import logging
import types
from functools import reduce

from visadore import base
import pyvisa
from tqdm import tqdm

from .api_types import XScale
from .api_types import YScale
from .api_types import WaveformCollection
from .api_types import Waveform
from ._tek_series_mso import WaveType
from ._tek_series_mso import JobParameters
from ._tek_series_mso import get_event_queue
from ._pyvisa_tqdm_patch import _read_raw_progress_bar
from ._pyvisa_tqdm_patch import read_binary_values_progress_bar
from ._pyvisa_tqdm_patch import read_bytes_progress_bar

log = logging.getLogger(__name__)

def _parse_sources_from_select_query(source_text):
    """Take the list of sources returned from a `SELECT?` query with header and verbose set to ON and return a list of channels"""
    # Format is something like: :SELECT:DIGGRP1:DALL 0;:SELECT:DIGGRP2:DALL 0;:SELECT:DIGGRP3:DALL 0;:SELECT:DIGGRP4:DALL 0;:SELECT:DIGGRP5:DALL 0;:SELECT:DIGGRP6:DALL 0;:SELECT:DIGGRP7:DALL 0;:SELECT:DIGGRP8:DALL 1;:SELECT:CH1 1;CH2 0;:SELECT:CH3 0;CH4 0;:SELECT:CH5 0;CH6 0;:SELECT:CH7 1;CH8 0
    source_text = source_text.replace(':SELECT:', '')
    all_channels = source_text.split(';')
    enabled_channels = []
    for i in range(len(all_channels)):
        name, enabled = all_channels[i].split(' ')
        if not enabled == '1':
            continue
        if name.startswith('DIGGRP'):
            assert len(name) == 12, f"Name of digital group '{name}' is not 12 characters long - we rely on this exactly"
            name = "CH"+name[6]+"_DALL"
        enabled_channels.append(name)                
    enabled_channels.sort()
    return enabled_channels


class TekSeriesCurveFeat(base.FeatureBase):
    def feature(self, *, use_pbar=True, decompose_dch=True, verbose=False):
        """
        Returns a WaveformCollection object containing waveform data available on the
        instrument.

        Parameters:
            use_pbar (bool): Optionally display a progress bar. (default: False)
            decompose_dch (bool): Optionally convert a DCH channel into eight separate
                1-bit channels. (default: True)
            verbose (bool): Display additional information
        """
        with self.resource_manager.open_resource(self.resource_name) as inst:
            result = WaveformCollection()

            # iterate through all available sources
            try:
                for ch, ch_data, x_scale, y_scale in self._get_data(
                    inst, self._list_sources(inst), use_pbar, decompose_dch
                ):
                    if verbose:
                        log.debug(ch)
                    result.data[ch] = Waveform(ch_data, x_scale, y_scale)
            except pyvisa.errors.VisaIOError:
                get_event_queue(inst)
                raise
        return result

    @staticmethod
    def _classify_waveform(source):
        if source[0:4] == "MATH":
            wave_type = WaveType.MATH
        elif "_" in source:
            wave_type = WaveType.DIGITAL
        else:
            wave_type = WaveType.ANALOG
        return wave_type

    @staticmethod
    def _has_data_available(instr, source):
        """Checks that the source seems to have actual data available for download.
        This function should be called before performing a curve query."""
        # Use "display:global:CH{x}:state?" to determine if the channel is displayed
        # and available for download
        if source == "NONE":
            display_on = False
        else:
            if source.endswith("_DALL"):
                source=source[:-5]
            display_on = bool(
                int(instr.query("display:global:{}:state?".format(source)))
            )
            # log.debug(source, display_on)
        return display_on

    @staticmethod
    def _get_xscale(instr):
        """Get the horizontal scale of the instrument"""

        # This function will generate a VISA timeout if there is no waveform data
        # available and the channel is enabled.
        try:
            xincr = instr.query("WFMOutpre:XINCR?").strip()
        except pyvisa.errors.VisaIOError:

            # Flush out the VISA interface event queue
            get_event_queue(instr, verbose=False)
            result = None
        else:

            # collect more horizontal data
            pt_off = instr.query("WFMOutpre:PT_OFF?").strip()
            xzero = instr.query("WFMOutpre:XZERO?").strip()
            xunit = instr.query("WFMOutpre:XUNIT?").strip()

            # calculate horizontal scale
            slope = float(xincr)
            offset = float(pt_off) * -slope + float(xzero)
            unit = xunit.strip('"')
            result = XScale(slope, offset, unit)
        return result

    @staticmethod
    def _get_yscale(instr, source):
        scale = float(instr.query("{}:SCALE?".format(source)))
        position = float(instr.query("{}:POSITION?".format(source)))
        top = scale * (5 - position)
        bottom = scale * (-5 - position)
        return YScale(top=top, bottom=bottom)


    def _list_sources(self, instr):
        """Returns a list of channel name strings that we could query"""

        # data:source:available does not return the digital channels on this scope, so we have to parse from a select query
        # log.debug("Using select2")
        instr.write("verbose ON;header ON")
        result_string = instr.query("select?")
        instr.write("verbose OFF;header OFF")
        available_waveforms = _parse_sources_from_select_query(result_string)
        # log.debug(f"   Wfm:{available_waveforms}")
        # Only include channels that available for download
        useful_channels = [
            i for i in available_waveforms if self._has_data_available(instr, i)
        ]
        # log.debug(f"Useful:{useful_channels}")

        # Return only waveforms that are available and have a corresponding useful
        # channel
        useful_waveforms = [
            i for i in available_waveforms if i in useful_channels
        ]
        # log.debug(f"UseWfm{useful_waveforms}")
        return useful_waveforms

    def _make_jobs(self, instr, sources):
        """Scan the instrument for available sources of data construct a list of jobs"""

        encoding_table = {
            WaveType.MATH: ("FPBinary", 16, "f"),
            WaveType.DIGITAL: ("RIBinary", 16, "h"),
            WaveType.ANALOG: ("RIBinary", 16, "h"),
        }
        channel_table = {
            WaveType.MATH: lambda x: x,
            WaveType.DIGITAL: lambda x: "_".join([x.split("_")[0], "DALL"]),
            WaveType.ANALOG: lambda x: x,
        }

        results = {}
        for source in sources:
            if source.split("_")[0] not in results:
                # Determine the type of waveform and set key interface parameters
                wave_type = self._classify_waveform(source)
                channel = channel_table[wave_type](source)
                encoding, bit_nr, datatype = encoding_table[wave_type]

                # Set the start and stop point of the record
                rec_len = int(instr.query("horizontal:recordlength?").strip())

                # Keep track of each super channel and math source that has been handled
                results[source.split("_")[0]] = JobParameters(
                    wave_type, channel, encoding, bit_nr, datatype, rec_len
                )

        return results

    @staticmethod
    def _setup_curve_query(instr, source, parameters: dict[str, JobParameters]):
        """Setup the instrument for the curve query operation"""

        # extract the job parameters
        wave_type, channel, encoding, bit_nr, datatype, rec_len = parameters[source]

        # Switch to the source and setup the data encoding
        instr.write("data:source {}".format(channel))
        instr.write("data:encdg {}".format(encoding))
        instr.write("WFMOUTPRE:BIT_NR {}".format(bit_nr))

        # Set the start and stop point of the record
        rec_len = instr.query("horizontal:recordlength?").strip()
        instr.write("data:start 1")
        instr.write("data:stop {}".format(rec_len))

    def _post_process_analog(self, instr, source, source_data, x_scale):
        """Post processes analog channel data"""

        # Normal analog channels must have the vertical scale and offset applied
        offset = float(instr.query("WFMOutpre:YZEro?"))
        scale = float(instr.query("WFMOutpre:YMUlt?"))
        source_data = [scale * i + offset for i in source_data]

        # Include y-scale information with analog channel waveforms
        y_scale = self._get_yscale(instr, source)

        return source, source_data, x_scale, y_scale

    @staticmethod
    def _post_process_digital_bits(sources, source, source_data, x_scale, bit):
        """Post processes digital channel data as separate bits"""

        bit_channel = "{}_D{}".format(source.split("_")[0], bit)

        # if the bit channel is available, decompose the data
        if bit_channel in sources:
            bit_data = [(i >> (2 * bit)) & 1 for i in source_data]
            return bit_channel, bit_data, x_scale, None

    @staticmethod
    def _post_process_digital_byte(source, source_data, x_scale):
        """Post processes digital channel data as a byte"""
        digital = []
        for i in source_data:
            a = (
                (i & 0x4000) >> 7
                | (i & 0x1000) >> 6
                | (i & 0x400) >> 5
                | (i & 0x100) >> 4
                | (i & 0x40) >> 3
                | (i & 0x10) >> 2
                | (i & 0x4) >> 1
                | i & 0x1
            )
            digital.append(a)
        return source.split("_")[0], digital, x_scale, None

    def _get_data(self, instr, sources, use_pbar, decompose_dch):
        """Returns an iterator that yields the source data from the oscilloscope"""
        if decompose_dch:
            logging.warning("Are you sure you want decompose_dch? It seems to end up with no digital data at all.")
        jobs = self._make_jobs(instr, sources)
        pbar_disabled = not use_pbar

        # remember the state of the acquisition system and then stop acquiring waveforms
        acq_state = instr.query("ACQuire:STATE?").strip()
        instr.write("ACQuire:STATE STOP")

        # Calculate the total number of bytes of data to be downloaded from the
        # instrument
        bytes_per_sample = {"FPBinary": 4, "RIBinary": 2}
        total_bytes = reduce(
            lambda a, b: a + b,
            [bytes_per_sample[jobs[i].encoding] * jobs[i].record_length for i in jobs],
        )

        with tqdm(
            desc="Downloading",
            unit="B",
            total=total_bytes,
            disable=pbar_disabled,
            unit_scale=True,
        ) as t:

            # Patch the instr object with a custom methods that implement tqdm updates
            instr.progress_bar = t
            instr.read_bytes_progress_bar = types.MethodType(
                read_bytes_progress_bar, instr
            )
            instr._read_raw_progress_bar = types.MethodType(
                _read_raw_progress_bar, instr
            )
            instr.read_binary_values_progress_bar = types.MethodType(
                read_binary_values_progress_bar,
                instr,
            )

            for source in jobs:
                
                self._setup_curve_query(instr, source, jobs)

                # extract the job parameters
                wave_type, channel, encoding, bit_nr, datatype, rec_len = jobs[source]

                # Horizontal scale information
                x_scale = self._get_xscale(instr)
                if x_scale is not None:

                    # Issue the curve query command
                    instr.write("curv?")

                    # Read the waveform data sent by the instrument
                    source_data = instr.read_binary_values_progress_bar(
                        datatype=datatype, is_big_endian=True, expect_termination=True
                    )

                    if wave_type is WaveType.DIGITAL:

                        # Digital channel to be decomposed into separate bits
                        if decompose_dch:
                            for bit in range(8):
                                result = self._post_process_digital_bits(
                                    sources, source, source_data, x_scale, bit
                                )
                                if result:
                                    yield result

                        # Digital channel to be converted into an 8-bit word
                        else:
                            yield self._post_process_digital_byte(
                                source, source_data, x_scale
                            )

                    elif wave_type is WaveType.ANALOG:
                        yield self._post_process_analog(
                            instr, source, source_data, x_scale
                        )

                    elif wave_type is WaveType.MATH:
                        # Y-scale information for MATH channels is not supported at
                        # this time
                        yield source, source_data, x_scale, None

                    else:
                        raise Exception(
                            "It should have been impossible to execute this code"
                        )

        # Restore the acquisition state
        instr.write("ACQuire:STATE {}".format(acq_state))
