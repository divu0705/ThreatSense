"""
constants.py
NSL-KDD column names and the standard attack-category mapping used across
the NIDS literature (grouping ~40 specific attack labels into 4 families:
DoS, Probe, R2L, U2R). This mapping is what most published NSL-KDD papers
use, including the ones cited in your IEEE Access survey.
"""

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# Specific attack label -> attack family. Anything not listed here that isn't
# "normal" falls back to "Unknown" (this happens for a handful of rare test-set
# attacks not present in the training set — a great real talking point about
# generalization for your thesis).
ATTACK_CATEGORY_MAP = {
    "normal": "Normal",
    # DoS
    "neptune": "DoS", "smurf": "DoS", "teardrop": "DoS", "pod": "DoS",
    "land": "DoS", "back": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS", "worm": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L (remote-to-local)
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R (user-to-root)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

SEVERITY_BY_CATEGORY = {
    "Normal": "none",
    "Probe": "low",
    "DoS": "high",
    "R2L": "medium",
    "U2R": "critical",
    "Unknown": "medium",
}
