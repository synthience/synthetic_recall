import asyncio
import logging
import ntplib
import time
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ClockDrift:
    start_time: float
    end_time: float
    drift_ms: float
    samples: int
    rtp_clock_rate: float

@dataclass
class RTPStats:
    """Statistics for RTP stream synchronization."""
    ssrc: int
    last_rtp_ts: int = 0
    last_ntp_ts: float = 0
    last_sr_ntp: Optional[float] = None
    last_sr_rtp: Optional[int] = None
    drift_samples: List[Dict] = field(default_factory=list)
    rtt_samples: List[float] = field(default_factory=list)
    jitter: float = 0.0
    packets_received: int = 0
    packets_lost: int = 0

class ClockSync:
    """Enhanced clock synchronization with RTP drift compensation."""
    
    def __init__(self, ntp_servers=None, sync_interval=30, sample_rate: int = 48000):
        self.ntp_client = ntplib.NTPClient()
        self.ntp_servers = ntp_servers or ['pool.ntp.org', 'time.google.com', 'time.windows.com']
        self.clock_offset = 0.0
        self.rtp_stats: Dict[int, RTPStats] = {}
        self.last_sync = 0
        self.sync_interval = sync_interval
        self.max_history = 20
        self.logger = logging.getLogger(__name__)
        self._consecutive_failures = 0
        self.offset_history = []
        self.sample_rate = sample_rate
        self.last_system_sync = 0
        self.system_offset = 0.0
        self.max_drift_samples = 50
        self.drift_correction_threshold = 10
        
    async def check_system_time(self) -> bool:
        """Check if system time is properly synchronized."""
        try:
            # Get Windows Time service status
            process = await asyncio.create_subprocess_shell(
                'w32tm /query /status',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"❌ Failed to query time status: {stderr.decode()}")
                return False
                
            output = stdout.decode()
            
            # Check if time is synced
            if "Last Successful Sync Time" in output:
                sync_time_str = output.split("Last Successful Sync Time: ")[1].split("\n")[0].strip()
                try:
                    sync_time = datetime.strptime(sync_time_str, "%d/%m/%Y %H:%M:%S")
                    time_diff = abs((datetime.now() - sync_time).total_seconds())
                    
                    if time_diff > 3600:  # More than 1 hour since last sync
                        logger.warning(f"⚠️ System time sync is old: {time_diff:.1f} seconds ago")
                        return False
                        
                    logger.info(f"✅ System time in sync (last sync: {time_diff:.1f} seconds ago)")
                    return True
                except ValueError:
                    logger.error(f"❌ Failed to parse sync time: {sync_time_str}")
                    return False
            else:
                logger.error("❌ Could not find last sync time in status output")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking system time: {e}")
            return False
            
    async def sync(self):
        if time.time() - self.last_sync < self.sync_interval:
            return

        failed_servers = []
        for server in self.ntp_servers:
            try:
                response = await asyncio.to_thread(self.ntp_client.request, server, timeout=2)
                self.offset_history.append({'timestamp': time.time(), 'offset': response.offset, 'delay': response.delay, 'server': server})

                if len(self.offset_history) > self.max_history:
                    self.offset_history.pop(0)

                self.last_sync = time.time()
                self.logger.debug(f"Clock sync: offset={response.offset:.3f}ms, delay={response.delay:.3f}ms from {server}")
                self._consecutive_failures = 0
                break
            except Exception as e:
                failed_servers.append(f"{server}: {e}")

        if len(failed_servers) == len(self.ntp_servers):
            self.logger.error(f"All NTP sync attempts failed: {failed_servers}")
            self._consecutive_failures += 1
            
        if await self.check_system_time():
            self.last_system_sync = time.time()
            logger.info("✅ Clock sync verified")
        else:
            logger.warning("⚠️ System time may be out of sync")
            
    async def force_sync(self):
        self.logger.info("Forcing manual clock sync...")
        await self.sync()

    async def sync_loop(self):
        while True:
            try:
                jitter = self.sync_interval * 0.1 * (2 * np.random.random() - 1)
                await asyncio.sleep(self.sync_interval + jitter)
                await self.sync()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in clock sync loop: {e}")

    @property
    def offset(self):
        return self.get_current_offset()

    @property
    def is_synced(self):
        return (time.time() - self.last_sync) <= self.sync_interval

    async def sync_clock(self) -> float:
        """Synchronize with NTP servers and return clock offset."""
        now = time.time()
        if now - self.last_sync < self.sync_interval:
            return self.clock_offset
            
        offsets = []
        for server in self.ntp_servers:
            try:
                response = self.ntp_client.request(server, timeout=2)
                offset = response.offset
                offsets.append(offset)
                logger.debug(f"🕒 NTP offset from {server}: {offset:.6f}s")
            except Exception as e:
                logger.warning(f"⚠️ Failed to sync with NTP server {server}: {e}")
                continue
                
        if offsets:
            # Use median to filter out outliers
            self.clock_offset = statistics.median(offsets)
            self.last_sync = now
            logger.info(f"✅ Clock synchronized, offset: {self.clock_offset:.6f}s")
        else:
            logger.error("❌ Failed to sync with any NTP servers")
            
        return self.clock_offset
        
    def get_corrected_time(self) -> float:
        """Get current time corrected by NTP offset."""
        return time.time() + self.clock_offset
        
    def update_rtp_stats(self, ssrc: int, rtp_ts: int, ntp_ts: float, sr_ntp: Optional[float] = None, sr_rtp: Optional[int] = None, rtt: Optional[float] = None):
        """Update RTP statistics with enhanced drift tracking."""
        if ssrc not in self.rtp_stats:
            self.rtp_stats[ssrc] = RTPStats(ssrc=ssrc)
            
        stats = self.rtp_stats[ssrc]
        stats.packets_received += 1
        
        # Update sender report info
        if sr_ntp is not None and sr_rtp is not None:
            stats.last_sr_ntp = sr_ntp
            stats.last_sr_rtp = sr_rtp
            
        # Calculate drift if we have previous samples
        if stats.last_rtp_ts > 0 and stats.last_ntp_ts > 0:
            rtp_diff = rtp_ts - stats.last_rtp_ts
            ntp_diff = (ntp_ts - stats.last_ntp_ts) * self.sample_rate
            
            if rtp_diff > 0 and ntp_diff > 0:
                # Calculate drift in samples
                drift = abs(rtp_diff - ntp_diff)
                # Calculate jitter (RFC 3550)
                stats.jitter = stats.jitter + (drift - stats.jitter) / 16
                
                stats.drift_samples.append({
                    'drift': drift,
                    'timestamp': ntp_ts,
                    'rtp_diff': rtp_diff,
                    'ntp_diff': ntp_diff
                })
                
                # Keep drift samples bounded
                while len(stats.drift_samples) > self.max_drift_samples:
                    stats.drift_samples.pop(0)
                    
        # Update RTT samples if provided
        if rtt is not None:
            stats.rtt_samples.append(rtt)
            while len(stats.rtt_samples) > self.max_rtt_samples:
                stats.rtt_samples.pop(0)
                
        # Update last seen timestamps
        stats.last_rtp_ts = rtp_ts
        stats.last_ntp_ts = ntp_ts
        
    def compensate_rtp_timestamp(self, rtp_timestamp: int, ssrc: int) -> int:
        """Compensate for RTP drift using weighted average of recent samples."""
        if ssrc not in self.rtp_stats:
            return rtp_timestamp
            
        stats = self.rtp_stats[ssrc]
        recent_drifts = stats.drift_samples[-5:]  # Use last 5 samples
        
        if not recent_drifts:
            return rtp_timestamp
            
        # Calculate weighted average (more recent samples have higher weight)
        total_weight = sum(range(1, len(recent_drifts) + 1))
        weighted_drift = sum(
            s['drift'] * (i + 1) for i, s in enumerate(recent_drifts)
        ) / total_weight
        
        # Apply compensation with jitter consideration
        compensation = int(weighted_drift + stats.jitter)
        compensated_ts = rtp_timestamp - compensation
        
        if compensation > 0:
            logger.debug(f"🔄 RTP compensation: {compensation} samples (drift: {weighted_drift:.1f}, jitter: {stats.jitter:.1f})")
            
        return compensated_ts
        
    def get_rtt_stats(self, ssrc: int) -> Optional[Dict[str, float]]:
        """Get RTT statistics for a stream."""
        if ssrc not in self.rtp_stats:
            return None
            
        stats = self.rtp_stats[ssrc]
        if not stats.rtt_samples:
            return None
            
        return {
            'min_rtt': min(stats.rtt_samples),
            'max_rtt': max(stats.rtt_samples),
            'avg_rtt': statistics.mean(stats.rtt_samples),
            'median_rtt': statistics.median(stats.rtt_samples)
        }
        
    def get_stream_stats(self, ssrc: int) -> Optional[Dict]:
        """Get comprehensive statistics for a stream."""
        if ssrc not in self.rtp_stats:
            return None
            
        stats = self.rtp_stats[ssrc]
        return {
            'packets_received': stats.packets_received,
            'packets_lost': stats.packets_lost,
            'jitter': stats.jitter,
            'rtt_stats': self.get_rtt_stats(ssrc),
            'last_drift': stats.drift_samples[-1] if stats.drift_samples else None
        }
        
    def clear_stats(self, ssrc: int):
        """Clear statistics for a stream."""
        if ssrc in self.rtp_stats:
            del self.rtp_stats[ssrc]
            logger.debug(f"🧹 Cleared RTP stats for SSRC {ssrc}")
            
    def clear_all_stats(self):
        """Clear all stream statistics."""
        self.rtp_stats.clear()
        logger.info("🧹 Cleared all RTP stats")


@dataclass
class SenderReport:
    ntp_timestamp: float
    rtp_timestamp: int
    packet_count: int
    octet_count: int
    received_at: float

@dataclass
class StreamStats:
    ssrc: str
    last_seq: int
    highest_seq: int
    cycle_count: int
    packets_received: int
    packets_lost: int
    last_sr_time: float
    last_sr_ntp: float
    last_sr_rtp: int
    jitter: float
    rtt: float

class EnhancedClockSync:
    """Enhanced clock synchronization with improved NTP and RTT handling."""
    
    def __init__(self):
        self._streams: Dict[str, StreamStats] = {}
        self._sender_reports: Dict[str, SenderReport] = {}
        self._ntp_offset: float = 0
        self._last_ntp_sync = 0
        self._sync_interval = 30  # Reduced from 60s to 30s
        self._max_rtt = 0.5      # Reduced from 1.0s to 500ms
        self._min_sr_interval = 0.1  # Reduced from 0.5s to 100ms
        self._last_sr_sent: Dict[str, float] = {}
        self._buffer_target = 50  # Target buffer size in milliseconds
        self._max_buffer = 100    # Maximum buffer size in milliseconds
        self._min_buffer = 20     # Minimum buffer size in milliseconds
        self._adaptive_buffer = True
        
    def _ntp_to_unix(self, ntp_timestamp: int) -> float:
        """Convert NTP timestamp to UNIX timestamp with microsecond precision."""
        ntp_era = (ntp_timestamp >> 32) & 0xFFFFFFFF
        ntp_secs = ntp_timestamp & 0xFFFFFFFF
        
        # NTP epoch starts at 1900, Unix at 1970
        ntp_to_unix_offset = 2208988800
        
        unix_secs = ntp_era - ntp_to_unix_offset
        unix_fraction = (ntp_secs * 1000000) >> 32  # Convert to microseconds
        
        return unix_secs + (unix_fraction / 1000000.0)
        
    def _unix_to_ntp(self, unix_timestamp: float) -> int:
        """Convert UNIX timestamp to NTP timestamp."""
        ntp_to_unix_offset = 2208988800
        
        ntp_secs = int(unix_timestamp) + ntp_to_unix_offset
        ntp_frac = int((unix_timestamp % 1.0) * 0x100000000)
        
        return (ntp_secs << 32) | ntp_frac
        
    def update_ntp_offset(self):
        """Update NTP offset with system time."""
        now = time.time()
        if now - self._last_ntp_sync >= self._sync_interval:
            try:
                # Get current UTC time
                utc_now = datetime.now(timezone.utc).timestamp()
                self._ntp_offset = utc_now - now
                self._last_ntp_sync = now
                logger.debug(f"Updated NTP offset: {self._ntp_offset*1000:.2f}ms")
            except Exception as e:
                logger.error(f"Failed to update NTP offset: {e}")
                
    def process_sender_report(self, ssrc: str, sr_data: Dict) -> Optional[float]:
        """Process RTCP sender report with improved timing accuracy."""
        try:
            now = time.time() + self._ntp_offset
            ntp_ts = sr_data.get('NtpTimestamp', 0)
            rtp_ts = sr_data.get('RtpTimestamp', 0)
            
            if not ntp_ts or not rtp_ts:
                return None
                
            # Convert NTP timestamp string to float
            if isinstance(ntp_ts, str):
                try:
                    ntp_dt = datetime.fromisoformat(ntp_ts.replace('Z', '+00:00'))
                    ntp_ts = ntp_dt.timestamp()
                except ValueError:
                    logger.error(f"Invalid NTP timestamp format: {ntp_ts}")
                    return None
                    
            # Check if we should process this SR
            if ssrc in self._last_sr_sent:
                time_since_last = now - self._last_sr_sent[ssrc]
                if time_since_last < self._min_sr_interval:
                    logger.debug(f"Skipping SR, too soon after last: {time_since_last*1000:.1f}ms")
                    return None
                    
            # Store sender report
            self._sender_reports[ssrc] = SenderReport(
                ntp_timestamp=ntp_ts,
                rtp_timestamp=rtp_ts,
                packet_count=sr_data.get('Packets', 0),
                octet_count=sr_data.get('Octets', 0),
                received_at=now
            )
            
            self._last_sr_sent[ssrc] = now
            
            # Calculate RTT if we have stream stats
            if ssrc in self._streams:
                stats = self._streams[ssrc]
                delay_since_last_sr = now - stats.last_sr_time
                
                if delay_since_last_sr > 0 and delay_since_last_sr < self._max_rtt:
                    rtt = delay_since_last_sr * 2  # Round trip is twice the one-way delay
                    stats.rtt = rtt
                    logger.debug(f"Updated RTT for stream {ssrc}: {rtt*1000:.1f}ms")
                    return rtt
                    
            return None
            
        except Exception as e:
            logger.error(f"Error processing sender report: {e}")
            return None
            
    def update_stream_stats(self, ssrc: str, seq: int, timestamp: int, received_at: float):
        """Update stream statistics with enhanced sequence tracking."""
        if ssrc not in self._streams:
            self._streams[ssrc] = StreamStats(
                ssrc=ssrc,
                last_seq=seq,
                highest_seq=seq,
                cycle_count=0,
                packets_received=1,
                packets_lost=0,
                last_sr_time=0,
                last_sr_ntp=0,
                last_sr_rtp=0,
                jitter=0,
                rtt=0
            )
            return
            
        stats = self._streams[ssrc]
        
        # Handle sequence number wrapping
        if seq < stats.last_seq and stats.last_seq - seq > 0x8000:
            stats.cycle_count += 1
            
        # Update highest sequence
        extended_seq = stats.cycle_count << 16 | seq
        if extended_seq > stats.highest_seq:
            stats.highest_seq = extended_seq
            
        # Calculate packets lost
        expected = stats.highest_seq - stats.last_seq + 1
        if expected > 0:
            stats.packets_lost = expected - stats.packets_received
            
        stats.last_seq = seq
        stats.packets_received += 1
        
        # Update jitter
        if ssrc in self._sender_reports:
            sr = self._sender_reports[ssrc]
            transit_time = (received_at - sr.received_at) - (
                (timestamp - sr.rtp_timestamp) / 48000  # Assuming 48kHz audio
            )
            delta = abs(transit_time - stats.jitter)
            stats.jitter = stats.jitter + (delta - stats.jitter) / 16
            
    def get_stream_stats(self, ssrc: str) -> Optional[Dict]:
        """Get stream statistics with enhanced metrics."""
        if ssrc not in self._streams:
            return None
            
        stats = self._streams[ssrc]
        return {
            'ssrc': stats.ssrc,
            'packets_received': stats.packets_received,
            'packets_lost': stats.packets_lost,
            'loss_rate': stats.packets_lost / max(stats.packets_received, 1),
            'jitter': stats.jitter * 1000,  # Convert to ms
            'rtt': stats.rtt * 1000,  # Convert to ms
            'last_seq': stats.last_seq,
            'highest_seq': stats.highest_seq
        }
        
    def estimate_rtp_timestamp(self, ssrc: str, ntp_time: float) -> Optional[int]:
        """Estimate RTP timestamp for a given NTP time with improved accuracy."""
        if ssrc not in self._sender_reports:
            return None
            
        sr = self._sender_reports[ssrc]
        time_diff = ntp_time - sr.ntp_timestamp
        
        # Convert time difference to RTP timestamp units (assuming 48kHz audio)
        timestamp_diff = int(time_diff * 48000)
        
        # Apply timestamp wraparound handling
        estimated_ts = (sr.rtp_timestamp + timestamp_diff) & 0xFFFFFFFF
        
        return estimated_ts
