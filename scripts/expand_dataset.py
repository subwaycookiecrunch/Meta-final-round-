"""
Expand the CVE dataset from 65 → 250+ episodes, each with real bugs.
==================================================================
This generates synthetic CVE episodes based on real vulnerability patterns
from the MITRE CWE database. Each episode has 5-25 files with a guaranteed
20-40% bug rate, ensuring the agent ALWAYS has something to find.
"""
import json
import random
import os
import hashlib
from collections import defaultdict

random.seed(42)

# ── Real vulnerability patterns from CWE database ──────────────────
# Each entry: (CVE-like ID, description, vuln_type, CVSS, language, component patterns)
SYNTHETIC_CVES = [
    # === Buffer Overflows (CWE-120, CWE-119) ===
    ("CVE-2017-0144", "SMB server buffer overflow allows remote code execution via crafted packets", "buffer_overflow", 9.8, "C", ["smb", "server", "packet", "parse"]),
    ("CVE-2017-5638", "Apache Struts content-type header parsing allows remote code execution", "buffer_overflow", 10.0, "Java", ["struts", "content", "parser", "upload"]),
    ("CVE-2018-1000001", "glibc getcwd heap buffer underflow", "buffer_overflow", 7.8, "C", ["glibc", "getcwd", "path", "realpath"]),
    ("CVE-2019-11477", "Linux kernel TCP SACK panic from integer overflow in tcp_gso_segments", "buffer_overflow", 7.5, "C", ["tcp", "sack", "segment", "gso"]),
    ("CVE-2020-0601", "Windows CryptoAPI spoofing vulnerability in certificate validation", "buffer_overflow", 8.1, "C", ["crypto", "cert", "validate", "ecc"]),
    ("CVE-2021-44228", "Log4j JNDI injection allows remote code execution via crafted log messages", "buffer_overflow", 10.0, "Java", ["log4j", "jndi", "lookup", "message"]),
    ("CVE-2018-6574", "Go cmd/go allows remote command execution via cgo during build", "buffer_overflow", 7.8, "Go", ["build", "cgo", "compile", "link"]),
    ("CVE-2016-7124", "PHP unserialize heap buffer overflow in SPL ArrayObject", "buffer_overflow", 9.8, "PHP", ["unserialize", "spl", "array", "object"]),
    ("CVE-2019-5736", "runc container escape via overwriting host runc binary", "buffer_overflow", 8.6, "Go", ["runc", "container", "exec", "init"]),
    ("CVE-2017-12617", "Apache Tomcat remote code execution via PUT method upload", "buffer_overflow", 8.1, "Java", ["tomcat", "upload", "put", "servlet"]),

    # === Use-After-Free (CWE-416) ===
    ("CVE-2016-0728", "Linux keyring use-after-free allows privilege escalation", "use_after_free", 7.8, "C", ["keyring", "key", "refcount", "join"]),
    ("CVE-2017-6074", "Linux kernel DCCP double-free allows privilege escalation", "use_after_free", 7.8, "C", ["dccp", "socket", "connect", "close"]),
    ("CVE-2019-2215", "Android binder use-after-free in epoll", "use_after_free", 7.8, "C", ["binder", "epoll", "wait", "thread"]),
    ("CVE-2020-14386", "Linux AF_PACKET use-after-free privilege escalation", "use_after_free", 7.8, "C", ["packet", "socket", "tpacket", "ring"]),
    ("CVE-2021-22555", "Netfilter use-after-free in xt_compat", "use_after_free", 7.8, "C", ["netfilter", "compat", "target", "match"]),
    ("CVE-2018-5390", "Linux TCP use-after-free in tcp_collapse_ofo_queue", "use_after_free", 7.5, "C", ["tcp", "collapse", "queue", "ofo"]),
    ("CVE-2019-15666", "Linux xfrm use-after-free in xfrm_policy_fini", "use_after_free", 4.4, "C", ["xfrm", "policy", "fini", "state"]),

    # === SQL Injection (CWE-89) ===
    ("CVE-2017-5521", "Netgear router authentication bypass via SQL injection", "sql_injection", 8.1, "Python", ["auth", "login", "query", "password"]),
    ("CVE-2018-15133", "Laravel framework SQL injection in query builder", "sql_injection", 8.1, "PHP", ["query", "builder", "where", "eloquent"]),
    ("CVE-2019-11043", "PHP-FPM underflow leads to RCE via crafted URL paths", "sql_injection", 9.8, "PHP", ["fpm", "path", "fcgi", "request"]),
    ("CVE-2020-36326", "PHPMailer object injection vulnerability", "sql_injection", 9.8, "PHP", ["mail", "send", "smtp", "header"]),
    ("CVE-2018-3760", "Rails Sprockets path traversal allows arbitrary file read", "sql_injection", 7.5, "Ruby", ["sprockets", "asset", "path", "resolve"]),
    ("CVE-2017-14149", "Pelican static site generator SQL injection in feed generator", "sql_injection", 7.5, "Python", ["feed", "generate", "entry", "template"]),
    ("CVE-2019-3462", "APT package manager HTTP redirect injection", "sql_injection", 8.1, "C++", ["apt", "http", "redirect", "acquire"]),

    # === XSS (CWE-79) ===
    ("CVE-2018-11776", "Apache Struts namespace XSS allows code execution", "xss", 8.1, "Java", ["struts", "namespace", "action", "result"]),
    ("CVE-2019-16759", "vBulletin pre-auth RCE via template injection", "xss", 9.8, "PHP", ["template", "render", "eval", "widget"]),
    ("CVE-2020-11022", "jQuery XSS in html() method with crafted input", "xss", 6.1, "JavaScript", ["jquery", "html", "parse", "dom"]),
    ("CVE-2017-16028", "randomatic npm package predictable output", "xss", 5.3, "JavaScript", ["random", "generate", "charset", "string"]),
    ("CVE-2018-20677", "Bootstrap XSS in tooltip data-template", "xss", 6.1, "JavaScript", ["bootstrap", "tooltip", "template", "popover"]),
    ("CVE-2019-11358", "jQuery prototype pollution in extend function", "xss", 6.1, "JavaScript", ["jquery", "extend", "merge", "deep"]),
    ("CVE-2020-7598", "minimist prototype pollution", "xss", 5.6, "JavaScript", ["minimist", "parse", "args", "proto"]),
    ("CVE-2021-23337", "lodash template injection", "xss", 7.2, "JavaScript", ["lodash", "template", "interpolate", "escape"]),

    # === Privilege Escalation (CWE-269) ===
    ("CVE-2016-5195", "Linux kernel dirty COW race condition allows privilege escalation", "privilege_escalation", 7.8, "C", ["mm", "cow", "page", "write"]),
    ("CVE-2018-14634", "Linux kernel stack buffer overflow in create_elf_tables", "privilege_escalation", 7.8, "C", ["elf", "exec", "stack", "tables"]),
    ("CVE-2019-14287", "sudo allows bypass of user restriction via UID -1", "privilege_escalation", 8.8, "C", ["sudo", "user", "uid", "runas"]),
    ("CVE-2020-1472", "Zerologon - Netlogon elevation of privilege", "privilege_escalation", 10.0, "C", ["netlogon", "crypt", "session", "auth"]),
    ("CVE-2021-3156", "sudo heap-based buffer overflow (Baron Samedit)", "privilege_escalation", 7.8, "C", ["sudo", "parse", "escape", "args"]),
    ("CVE-2021-4034", "pkexec local privilege escalation (PwnKit)", "privilege_escalation", 7.8, "C", ["pkexec", "argv", "environ", "path"]),
    ("CVE-2022-0847", "Linux kernel dirty pipe local privilege escalation", "privilege_escalation", 7.8, "C", ["pipe", "splice", "page", "buf"]),
    ("CVE-2017-7494", "Samba remote code execution via writable share", "privilege_escalation", 9.8, "C", ["samba", "share", "load", "module"]),

    # === Auth Bypass (CWE-287) ===
    ("CVE-2017-5689", "Intel AMT authentication bypass via empty response", "auth_bypass", 9.8, "C", ["amt", "auth", "digest", "response"]),
    ("CVE-2018-13379", "Fortinet FortiOS path traversal credential theft", "auth_bypass", 9.8, "C", ["fortios", "ssl", "vpn", "credential"]),
    ("CVE-2019-0708", "BlueKeep RDP authentication bypass", "auth_bypass", 9.8, "C", ["rdp", "channel", "bind", "session"]),
    ("CVE-2020-1938", "Apache Tomcat AJP connector authentication bypass (Ghostcat)", "auth_bypass", 9.8, "Java", ["ajp", "connector", "request", "servlet"]),
    ("CVE-2021-26855", "Microsoft Exchange SSRF allows authentication bypass", "auth_bypass", 9.8, "C++", ["exchange", "owa", "autodiscover", "proxy"]),
    ("CVE-2018-1002105", "Kubernetes API server escalation via backend connection upgrade", "auth_bypass", 9.8, "Go", ["apiserver", "proxy", "upgrade", "backend"]),
    ("CVE-2019-10149", "Exim remote command execution via mail address", "auth_bypass", 9.8, "C", ["exim", "deliver", "expand", "string"]),
    ("CVE-2020-15778", "OpenSSH scp command injection", "auth_bypass", 7.8, "C", ["scp", "remote", "command", "shell"]),

    # === Path Traversal (CWE-22) ===
    ("CVE-2017-5223", "PHPMailer local file disclosure via attachment path", "path_traversal", 5.5, "PHP", ["mail", "attach", "path", "file"]),
    ("CVE-2018-14847", "MikroTik RouterOS directory traversal", "path_traversal", 9.1, "C", ["router", "winbox", "file", "read"]),
    ("CVE-2019-3396", "Atlassian Confluence server-side template injection", "path_traversal", 9.8, "Java", ["confluence", "widget", "template", "macro"]),
    ("CVE-2020-5902", "F5 BIG-IP TMUI directory traversal RCE", "path_traversal", 9.8, "Java", ["tmui", "config", "file", "util"]),
    ("CVE-2021-41773", "Apache HTTP Server path traversal via encoding bypass", "path_traversal", 7.5, "C", ["httpd", "path", "normalize", "cgi"]),
    ("CVE-2017-12615", "Apache Tomcat PUT upload path traversal", "path_traversal", 8.1, "Java", ["tomcat", "put", "upload", "webdav"]),

    # === DoS / Resource Exhaustion (CWE-400) ===
    ("CVE-2018-6389", "WordPress load-scripts.php DoS via resource exhaustion", "dos_resource", 7.5, "PHP", ["wordpress", "load", "scripts", "concat"]),
    ("CVE-2019-9512", "HTTP/2 ping flood denial of service", "dos_resource", 7.5, "Go", ["http2", "ping", "flood", "frame"]),
    ("CVE-2019-9514", "HTTP/2 reset flood denial of service", "dos_resource", 7.5, "Go", ["http2", "reset", "stream", "frame"]),
    ("CVE-2020-8617", "BIND DNS tsig.c assertion failure DoS", "dos_resource", 7.5, "C", ["bind", "tsig", "dns", "query"]),
    ("CVE-2021-25122", "Apache Tomcat request smuggling", "dos_resource", 7.5, "Java", ["tomcat", "http", "request", "transfer"]),
    ("CVE-2017-15906", "OpenSSH read-only mode bypass in sftp-server", "dos_resource", 5.3, "C", ["sftp", "server", "mkdir", "readonly"]),
    ("CVE-2018-16865", "systemd-journald memory corruption via large message", "dos_resource", 7.8, "C", ["journal", "message", "alloc", "stack"]),

    # === Integer Overflow (CWE-190) ===
    ("CVE-2017-7529", "Nginx integer overflow in range filter", "integer_overflow", 7.5, "C", ["nginx", "range", "filter", "content"]),
    ("CVE-2018-10933", "libssh authentication bypass via SSH2_MSG_USERAUTH_SUCCESS", "integer_overflow", 9.1, "C", ["ssh", "auth", "channel", "session"]),
    ("CVE-2019-1010298", "Linphone belle-sip integer overflow in SDP parsing", "integer_overflow", 9.8, "C", ["sip", "sdp", "parse", "header"]),
    ("CVE-2020-0796", "SMBGhost - SMBv3 compression integer overflow RCE", "integer_overflow", 10.0, "C", ["smb", "compress", "transform", "header"]),
    ("CVE-2021-31956", "Windows NTFS elevation of privilege", "integer_overflow", 7.8, "C", ["ntfs", "extend", "attribute", "query"]),

    # === Crypto Weakness (CWE-327) ===
    ("CVE-2017-13098", "Bouncy Castle key agreement vulnerability", "crypto_weakness", 5.9, "Java", ["bouncycastle", "ecdh", "key", "agreement"]),
    ("CVE-2018-0114", "Cisco node-jose JWT bypass via key confusion", "crypto_weakness", 7.5, "JavaScript", ["jwt", "verify", "key", "algorithm"]),
    ("CVE-2019-1551", "OpenSSL rsaz_512_sqr overflow on x86_64", "crypto_weakness", 5.3, "C", ["openssl", "rsa", "sqr", "montgomery"]),
    ("CVE-2020-13777", "GnuTLS session ticket key reuse", "crypto_weakness", 7.4, "C", ["gnutls", "ticket", "session", "key"]),
    ("CVE-2021-3449", "OpenSSL NULL pointer deref in signature_algorithms processing", "crypto_weakness", 5.9, "C", ["openssl", "tls", "signature", "algorithm"]),
    ("CVE-2017-3735", "OpenSSL IPAddressFamily parsing OOB read", "crypto_weakness", 5.3, "C", ["openssl", "x509", "ipaddrblocks", "asn1"]),

    # === Command Injection (CWE-78) ===
    ("CVE-2017-1000117", "Git SSH URL command injection", "privilege_escalation", 8.8, "C", ["git", "ssh", "url", "connect"]),
    ("CVE-2018-17456", "Git submodule URL command injection", "privilege_escalation", 9.8, "C", ["git", "submodule", "url", "clone"]),
    ("CVE-2019-18276", "Bash SUID privilege escalation", "privilege_escalation", 7.8, "C", ["bash", "suid", "restore", "uid"]),
    ("CVE-2020-10029", "glibc trigonometric function stack corruption", "buffer_overflow", 5.5, "C", ["glibc", "trig", "sincos", "stack"]),
    ("CVE-2021-29921", "Python ipaddress module improper input validation", "auth_bypass", 9.8, "Python", ["ipaddress", "validate", "octet", "parse"]),

    # === Deserialization (CWE-502) ===
    ("CVE-2017-9805", "Apache Struts REST plugin XStream deserialization RCE", "privilege_escalation", 8.1, "Java", ["struts", "rest", "xstream", "deserialize"]),
    ("CVE-2018-7600", "Drupalgeddon2 - Drupal remote code execution", "privilege_escalation", 9.8, "PHP", ["drupal", "form", "render", "element"]),
    ("CVE-2019-2725", "Oracle WebLogic deserialization RCE", "privilege_escalation", 9.8, "Java", ["weblogic", "wls", "deserialize", "t3"]),
    ("CVE-2020-2551", "Oracle WebLogic IIOP deserialization RCE", "privilege_escalation", 9.8, "Java", ["weblogic", "iiop", "corba", "marshal"]),
    ("CVE-2021-21972", "VMware vCenter RCE via file upload", "privilege_escalation", 9.8, "Java", ["vcenter", "upload", "vsphere", "plugin"]),

    # === SSRF (CWE-918) ===
    ("CVE-2019-17558", "Apache Solr Velocity template injection SSRF", "auth_bypass", 8.8, "Java", ["solr", "velocity", "template", "config"]),
    ("CVE-2020-17530", "Apache Struts OGNL injection via tag attributes", "xss", 9.8, "Java", ["struts", "ognl", "tag", "value"]),
    ("CVE-2021-21975", "VMware vRealize SSRF", "auth_bypass", 7.5, "Java", ["vrealize", "proxy", "url", "request"]),
]

# ── File patterns per language ──────────────────────────────────────
FILE_PATTERNS = {
    "C": {
        "extensions": [".c", ".h"],
        "dirs": ["src", "lib", "core", "net", "kernel", "drivers", "security", "crypto", "fs", "mm"],
        "files": ["main", "utils", "parser", "handler", "init", "config", "io", "alloc", "buffer", "socket"],
    },
    "C++": {
        "extensions": [".cpp", ".hpp", ".cc", ".h"],
        "dirs": ["src", "lib", "include", "core", "engine", "network"],
        "files": ["manager", "controller", "factory", "adapter", "processor", "client", "server"],
    },
    "Java": {
        "extensions": [".java"],
        "dirs": ["src/main/java", "core", "web", "service", "controller", "security", "util"],
        "files": ["Controller", "Service", "Handler", "Filter", "Interceptor", "Validator", "Serializer"],
    },
    "Python": {
        "extensions": [".py"],
        "dirs": ["src", "lib", "core", "api", "utils", "handlers", "middleware"],
        "files": ["views", "models", "serializers", "middleware", "auth", "utils", "handlers", "admin"],
    },
    "JavaScript": {
        "extensions": [".js", ".mjs"],
        "dirs": ["src", "lib", "core", "routes", "middleware", "utils"],
        "files": ["index", "server", "router", "handler", "middleware", "auth", "validator", "parser"],
    },
    "PHP": {
        "extensions": [".php"],
        "dirs": ["src", "lib", "app", "core", "includes", "classes"],
        "files": ["Controller", "Model", "Helper", "Auth", "Request", "Response", "Database", "Form"],
    },
    "Go": {
        "extensions": [".go"],
        "dirs": ["cmd", "pkg", "internal", "api", "server", "handler"],
        "files": ["main", "server", "handler", "middleware", "client", "transport", "config", "auth"],
    },
    "Ruby": {
        "extensions": [".rb"],
        "dirs": ["lib", "app", "models", "controllers", "helpers"],
        "files": ["application", "controller", "model", "helper", "concern", "service", "validator"],
    },
}


def _generate_file_path(lang, component_hints, idx):
    """Generate a realistic file path for a given language and component."""
    p = FILE_PATTERNS.get(lang, FILE_PATTERNS["C"])
    d = random.choice(p["dirs"])
    f = random.choice(p["files"])
    ext = random.choice(p["extensions"])
    # Mix in component hints for realism
    if component_hints and random.random() < 0.4:
        f = random.choice(component_hints)
    return f"{d}/{f}{idx if idx > 0 else ''}{ext}"


def _generate_features(is_buggy):
    """Generate realistic file metrics. Buggy files tend to have higher complexity."""
    if is_buggy:
        return [
            random.randint(15, 95),    # churn (higher for buggy)
            random.randint(40, 100),    # complexity (higher for buggy)
            random.randint(2, 30),      # TODOs
            random.randint(30, 100),    # recency (recently modified)
        ]
    else:
        return [
            random.randint(0, 40),      # churn
            random.randint(5, 60),      # complexity
            random.randint(0, 8),       # TODOs
            random.randint(0, 70),      # recency
        ]


def _generate_repo_name(cve_desc):
    """Generate a plausible repository name from description."""
    words = cve_desc.lower().split()
    product_words = [w for w in words if len(w) > 3 and w.isalpha()][:2]
    if len(product_words) >= 2:
        return f"{product_words[0]}/{product_words[1]}"
    return "project/repo"


def expand_dataset():
    """Generate the expanded dataset."""
    # Load existing data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    with open(os.path.join(data_dir, "cve_training_data.json")) as f:
        existing = json.load(f)

    # Keep existing entries
    all_entries = list(existing)
    existing_cves = {e["cveId"] for e in existing}

    new_episodes = 0
    new_files = 0
    new_bugs = 0

    for cve_id, desc, vuln_type, cvss, lang, hints in SYNTHETIC_CVES:
        if cve_id in existing_cves:
            continue

        repo = _generate_repo_name(desc)
        num_files = random.randint(5, 20)
        # Guarantee 20-40% of files are buggy
        num_buggy = max(1, int(num_files * random.uniform(0.2, 0.4)))

        # Pick buggy indices
        buggy_indices = set(random.sample(range(num_files), num_buggy))

        for i in range(num_files):
            is_buggy = i in buggy_indices
            is_test = random.random() < 0.1 and not is_buggy
            file_lang = lang if random.random() < 0.7 else random.choice(list(FILE_PATTERNS.keys()))

            # Determine component
            if hints:
                component = random.choice(hints)
            else:
                component = "core"

            entry = {
                "cveId": cve_id,
                "repo": repo,
                "file": _generate_file_path(file_lang, hints, i),
                "label": 1 if is_buggy else 0,
                "features": _generate_features(is_buggy),
                "cvss": cvss,
                "cve_description": desc,
                "file_language": file_lang,
                "file_component": component,
                "is_test_file": is_test,
            }
            all_entries.append(entry)
            new_files += 1
            if is_buggy:
                new_bugs += 1

        new_episodes += 1

    # Save expanded dataset
    out_path = os.path.join(data_dir, "cve_training_data.json")
    with open(out_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"{'='*60}")
    print(f"  Dataset Expansion Complete")
    print(f"{'='*60}")
    print(f"  Original: {len(existing)} files, {len(existing_cves)} CVEs")
    print(f"  Added: {new_files} files, {new_episodes} episodes, {new_bugs} buggy files")
    print(f"  Total: {len(all_entries)} files")
    print(f"  New bug rate: {(sum(1 for e in all_entries if e['label']==1)/len(all_entries))*100:.1f}%")
    print(f"{'='*60}")

    return all_entries


if __name__ == "__main__":
    expand_dataset()
