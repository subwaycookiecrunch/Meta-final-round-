"""
Generate realistic code snippets for all files in the CVE dataset.
Buggy files get vulnerability-matching code; clean files get safe code.
"""
import json
import random
import os
import re

# ── Vulnerability templates by CVE pattern ──────────────────────────
VULN_TEMPLATES = {
    "buffer_overflow": {
        "C": '''// {func_name} - processes user input
int {func_name}(const char *input, size_t len) {{
    char buf[256];
    int i, count = 0;
    
    // BUG: no bounds check on len vs buf size
    for (i = 0; i < len; i++) {{
        buf[i] = input[i];  // potential buffer overflow
        if (input[i] == '\\n') count++;
    }}
    buf[i] = '\\0';
    
    if (count > MAX_LINES) {{
        log_warning("excessive line count: %d", count);
        return -EINVAL;
    }}
    return process_buffer(buf, count);
}}''',
        "C++": '''// {func_name} - decode input stream
void {func_name}(const uint8_t *data, size_t size) {{
    uint8_t output[512];
    size_t out_pos = 0;
    
    for (size_t i = 0; i < size; i++) {{
        // BUG: out_pos not checked against output size
        if (data[i] == ESCAPE_BYTE) {{
            output[out_pos++] = decode_escape(data[++i]);
        }} else {{
            output[out_pos++] = data[i];
        }}
    }}
    emit_decoded(output, out_pos);
}}''',
    },
    "integer_overflow": {
        "C": '''// {func_name} - allocate resources
static int {func_name}(unsigned int num_items, unsigned int item_size) {{
    unsigned int total;
    void *buf;
    
    // BUG: integer overflow when num_items * item_size > UINT_MAX
    total = num_items * item_size;
    buf = kmalloc(total, GFP_KERNEL);
    if (!buf)
        return -ENOMEM;
    
    memset(buf, 0, total);
    return init_items(buf, num_items);
}}''',
    },
    "use_after_free": {
        "C": '''// {func_name} - handle object lifecycle
static void {func_name}(struct {obj_type} *obj) {{
    struct {obj_type} *ref = obj->parent;
    
    if (obj->flags & FLAG_PENDING) {{
        cancel_pending_work(obj);
    }}
    
    kfree(obj);  // object freed here
    
    // BUG: use after free - ref->child still points to freed obj
    if (ref && ref->child == obj) {{
        ref->child = NULL;  // accessing freed memory
        notify_parent(ref);
    }}
}}''',
    },
    "privilege_escalation": {
        "C": '''// {func_name} - handle compat ioctl
static int {func_name}(struct sock *sk, int cmd, void __user *arg) {{
    struct compat_data __user *compat = arg;
    struct native_data data;
    int ret;
    
    // BUG: insufficient validation of user-supplied size
    if (copy_from_user(&data, compat, sizeof(data)))
        return -EFAULT;
    
    // Missing capability check for privileged operation
    ret = do_privileged_operation(sk, &data);
    if (ret < 0)
        return ret;
    
    return copy_to_user(arg, &data, sizeof(data)) ? -EFAULT : 0;
}}''',
    },
    "xss": {
        "Python": '''# {func_name} - render user content
def {func_name}(request):
    title = request.GET.get("title", "")
    content = request.GET.get("content", "")
    
    # BUG: user input rendered without escaping
    html = f"<h1>{{title}}</h1><div>{{content}}</div>"
    return HttpResponse(html)
''',
        "JavaScript": '''// {func_name} - display user message
function {func_name}(userInput) {{
    const container = document.getElementById("output");
    // BUG: direct innerHTML assignment with unsanitized input
    container.innerHTML = "<div class='msg'>" + userInput + "</div>";
    updateTimestamp();
}}''',
        "PHP": '''// {func_name} - show search results
function {func_name}($query) {{
    $results = db_search($query);
    // BUG: unescaped user input in HTML output
    echo "<h2>Results for: " . $query . "</h2>";
    foreach ($results as $r) {{
        echo "<li>" . $r["title"] . "</li>";
    }}
}}''',
    },
    "sql_injection": {
        "Python": '''# {func_name} - search records
def {func_name}(search_term, category=None):
    # BUG: string interpolation in SQL query
    query = f"SELECT * FROM records WHERE name LIKE '%{{search_term}}%'"
    if category:
        query += f" AND category = '{{category}}'"
    
    cursor = db.execute(query)
    return cursor.fetchall()
''',
        "PHP": '''// {func_name} - fetch user data
function {func_name}($user_id) {{
    // BUG: unsanitized input in SQL query
    $query = "SELECT * FROM users WHERE id = " . $user_id;
    $result = mysqli_query($conn, $query);
    return mysqli_fetch_assoc($result);
}}''',
    },
    "dos_resource": {
        "C": '''// {func_name} - parse input data
int {func_name}(const uint8_t *data, size_t len) {{
    size_t pos = 0;
    int depth = 0;
    
    // BUG: no depth limit allows stack exhaustion
    while (pos < len) {{
        if (data[pos] == OPEN_TAG) {{
            depth++;
            parse_nested(data + pos, len - pos, depth);
        }}
        pos++;
    }}
    return 0;
}}''',
        "Python": '''# {func_name} - validate input
def {func_name}(data):
    import re
    # BUG: catastrophic backtracking on crafted input
    pattern = r"^(a+)+$"
    if re.match(pattern, data):
        return True
    return False
''',
        "Go": '''// {func_name} - process request
func {func_name}(r *http.Request) error {{
    body, err := ioutil.ReadAll(r.Body)
    if err != nil {{
        return err
    }}
    // BUG: no size limit on request body
    var data map[string]interface{{}}
    if err := json.Unmarshal(body, &data); err != nil {{
        return fmt.Errorf("invalid JSON: %w", err)
    }}
    return processData(data)
}}''',
    },
    "auth_bypass": {
        "Python": '''# {func_name} - verify authentication
def {func_name}(request):
    token = request.headers.get("Authorization", "")
    # BUG: weak token validation, timing-safe compare not used
    if token == EXPECTED_TOKEN:
        return True
    # BUG: fallback allows bypass via specific header
    if request.headers.get("X-Internal-Request") == "true":
        return True  # intended for internal use but externally accessible
    return False
''',
        "JavaScript": '''// {func_name} - check permissions
function {func_name}(user, resource) {{
    // BUG: prototype pollution can bypass this check
    if (user.role === "admin") return true;
    const perms = user.permissions || {{}};
    return perms[resource] === true;
}}''',
    },
    "path_traversal": {
        "Python": '''# {func_name} - serve file
def {func_name}(filename):
    # BUG: no path traversal protection
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "rb") as f:
        return f.read()
''',
    },
    "crypto_weakness": {
        "Go": '''// {func_name} - generate token
func {func_name}(userID string) string {{
    // BUG: weak random source for security token
    rand.Seed(time.Now().UnixNano())
    token := make([]byte, 32)
    for i := range token {{
        token[i] = byte(rand.Intn(256))
    }}
    return base64.StdEncoding.EncodeToString(token)
}}''',
        "Python": '''# {func_name} - create session token
def {func_name}(user_id):
    import random
    import hashlib
    # BUG: predictable random for security-sensitive token
    seed = str(random.random()) + str(user_id)
    return hashlib.md5(seed.encode()).hexdigest()
''',
    },
}

# ── Clean code templates by language ────────────────────────────────
CLEAN_TEMPLATES = {
    "C": [
'''// {func_name} - initialize component
static int {func_name}(struct {component} *ctx) {{
    if (!ctx)
        return -EINVAL;
    
    memset(ctx, 0, sizeof(*ctx));
    ctx->state = STATE_INIT;
    ctx->refcount = 1;
    
    spin_lock_init(&ctx->lock);
    INIT_LIST_HEAD(&ctx->entries);
    
    return 0;
}}''',
'''// {func_name} - cleanup resources  
static void {func_name}(struct {component} *ctx) {{
    if (!ctx)
        return;
    
    spin_lock(&ctx->lock);
    ctx->state = STATE_CLOSING;
    spin_unlock(&ctx->lock);
    
    flush_workqueue(ctx->wq);
    destroy_workqueue(ctx->wq);
    kfree(ctx);
}}''',
'''// {func_name} - lookup entry by id
static struct entry *{func_name}(struct {component} *ctx, uint32_t id) {{
    struct entry *e;
    
    rcu_read_lock();
    list_for_each_entry_rcu(e, &ctx->entries, node) {{
        if (e->id == id) {{
            rcu_read_unlock();
            return e;
        }}
    }}
    rcu_read_unlock();
    return NULL;
}}''',
    ],
    "C/C++ Header": [
'''#ifndef _{guard}_H
#define _{guard}_H

#include <stdint.h>
#include <stdbool.h>

struct {component}_config {{
    uint32_t max_entries;
    uint32_t timeout_ms;
    bool     enable_logging;
    void    *priv_data;
}};

int  {component}_init(struct {component}_config *cfg);
void {component}_cleanup(void);
int  {component}_process(const void *data, size_t len);

#endif /* _{guard}_H */''',
    ],
    "Python": [
'''# {func_name} - process entries
def {func_name}(entries, config=None):
    """Process a batch of entries with optional configuration."""
    config = config or {{}}
    results = []
    
    for entry in entries:
        if not _validate_entry(entry):
            logger.warning("Skipping invalid entry: %s", entry.get("id"))
            continue
        
        result = _transform_entry(entry, config)
        results.append(result)
    
    logger.info("Processed %d/%d entries", len(results), len(entries))
    return results
''',
'''# {func_name} - data validation
class {component}Validator:
    """Validates {component} data against schema."""
    
    def __init__(self, schema):
        self.schema = schema
        self._cache = {{}}
    
    def validate(self, data):
        errors = []
        for field, rules in self.schema.items():
            value = data.get(field)
            if rules.get("required") and value is None:
                errors.append(f"Missing required field: {{field}}")
            elif value is not None and not isinstance(value, rules["type"]):
                errors.append(f"Invalid type for {{field}}")
        return errors
''',
    ],
    "JavaScript": [
'''// {func_name} - event handler
function {func_name}(event) {{
    const target = event.target;
    if (!target || !target.dataset.action) return;
    
    const action = target.dataset.action;
    const id = parseInt(target.dataset.id, 10);
    
    switch (action) {{
        case "update":
            handleUpdate(id);
            break;
        case "delete":
            if (confirm("Are you sure?")) {{
                handleDelete(id);
            }}
            break;
        default:
            console.warn("Unknown action:", action);
    }}
}}''',
    ],
    "Go": [
'''// {func_name} handles the request
func {func_name}(w http.ResponseWriter, r *http.Request) {{
    if r.Method != http.MethodPost {{
        http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
        return
    }}
    
    var req Request
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {{
        http.Error(w, "invalid request", http.StatusBadRequest)
        return
    }}
    defer r.Body.Close()
    
    result, err := process(r.Context(), &req)
    if err != nil {{
        log.Printf("error processing request: %v", err)
        http.Error(w, "internal error", http.StatusInternalServerError)
        return
    }}
    
    json.NewEncoder(w).Encode(result)
}}''',
    ],
    "Ruby": [
'''# {func_name} - process request
def {func_name}(params)
  validate_params!(params)
  
  result = {{}}
  params.each do |key, value|
    result[key] = sanitize(value)
  end
  
  save_to_store(result)
  result
rescue ValidationError => e
  logger.error("Validation failed: #{{e.message}}")
  nil
end''',
    ],
    "Java": [
'''// {func_name} - service method
public ResponseEntity<Result> {func_name}(RequestDTO request) {{
    if (request == null || !request.isValid()) {{
        return ResponseEntity.badRequest().build();
    }}
    
    try {{
        Result result = service.process(request);
        return ResponseEntity.ok(result);
    }} catch (ServiceException e) {{
        log.error("Processing failed: {{}}", e.getMessage());
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
    }}
}}''',
    ],
    "PHP": [
'''// {func_name} - handle form submission
function {func_name}($request) {{
    $data = $request->validate([
        "name" => "required|string|max:255",
        "email" => "required|email",
    ]);
    
    $record = Model::create($data);
    return redirect()->route("records.show", $record->id)
        ->with("success", "Record created.");
}}''',
    ],
}

# Default for languages without specific templates
DEFAULT_CLEAN = '''// {func_name} - utility function
// Component: {component}
// Standard implementation following project conventions

function {func_name}(input) {{
    if (!input) return null;
    
    const result = transform(input);
    validate(result);
    
    return result;
}}'''

# ── CVE to vulnerability type mapping ───────────────────────────────
CVE_VULN_MAP = {
    "CVE-2012-0044": "integer_overflow",
    "CVE-2012-4792": "use_after_free",
    "CVE-2013-7448": "xss",
    "CVE-2014-3153": "privilege_escalation",
    "CVE-2014-3576": "dos_resource",
    "CVE-2014-9765": "buffer_overflow",
    "CVE-2014-9773": "auth_bypass",
    "CVE-2015-1789": "dos_resource",
    "CVE-2015-2912": "privilege_escalation",
    "CVE-2015-6925": "dos_resource",
    "CVE-2015-7551": "path_traversal",
    "CVE-2015-8080": "integer_overflow",
    "CVE-2015-8379": "auth_bypass",
    "CVE-2015-8400": "crypto_weakness",
    "CVE-2015-8474": "auth_bypass",
    "CVE-2015-8547": "dos_resource",
    "CVE-2015-8612": "privilege_escalation",
    "CVE-2015-8618": "crypto_weakness",
    "CVE-2015-8630": "dos_resource",
    "CVE-2015-8702": "buffer_overflow",
    "CVE-2015-8865": "dos_resource",
    "CVE-2015-8877": "dos_resource",
    "CVE-2016-0738": "dos_resource",
    "CVE-2016-1181": "privilege_escalation",
    "CVE-2016-1202": "path_traversal",
    "CVE-2016-1405": "buffer_overflow",
    "CVE-2016-1541": "buffer_overflow",
    "CVE-2016-1709": "buffer_overflow",
    "CVE-2016-1902": "crypto_weakness",
    "CVE-2016-1904": "integer_overflow",
    "CVE-2016-1905": "auth_bypass",
    "CVE-2016-1927": "xss",
    "CVE-2016-2052": "dos_resource",
    "CVE-2016-2160": "privilege_escalation",
    "CVE-2016-2183": "crypto_weakness",
    "CVE-2016-2510": "privilege_escalation",
    "CVE-2016-2512": "auth_bypass",
    "CVE-2016-2515": "dos_resource",
    "CVE-2016-2537": "dos_resource",
    "CVE-2016-3062": "buffer_overflow",
    "CVE-2016-3072": "sql_injection",
    "CVE-2016-3096": "path_traversal",
    "CVE-2016-3674": "path_traversal",
    "CVE-2016-3693": "auth_bypass",
    "CVE-2016-3697": "privilege_escalation",
    "CVE-2016-3698": "auth_bypass",
    "CVE-2016-3728": "privilege_escalation",
    "CVE-2016-4021": "dos_resource",
    "CVE-2016-4040": "sql_injection",
    "CVE-2016-4069": "auth_bypass",
    "CVE-2016-4074": "dos_resource",
    "CVE-2016-4309": "xss",
    "CVE-2016-4425": "dos_resource",
    "CVE-2016-4562": "dos_resource",
    "CVE-2016-4630": "buffer_overflow",
    "CVE-2016-4817": "dos_resource",
    "CVE-2016-4979": "auth_bypass",
    "CVE-2016-4997": "privilege_escalation",
    "CVE-2016-5157": "buffer_overflow",
    "CVE-2016-5301": "dos_resource",
    "CVE-2016-5350": "dos_resource",
    "CVE-2016-5361": "dos_resource",
    "CVE-2016-5385": "auth_bypass",
    "CVE-2016-5835": "auth_bypass",
    "CVE-2016-6515": "dos_resource",
}


def _func_from_path(filepath):
    """Generate a plausible function name from file path."""
    base = os.path.splitext(os.path.basename(filepath))[0]
    # Convert to snake_case
    name = re.sub(r'[^a-zA-Z0-9]', '_', base).lower()
    name = re.sub(r'_+', '_', name).strip('_')
    return name[:30] or "process"


def _component_from_path(filepath):
    parts = filepath.split('/')
    if len(parts) >= 2:
        return re.sub(r'[^a-zA-Z0-9]', '_', parts[-2]).lower()[:20]
    return "core"


def generate_vuln_snippet(file_entry, cve_id):
    """Generate vulnerable code for a buggy file."""
    lang = file_entry.get("file_language", "C")
    vuln_type = CVE_VULN_MAP.get(cve_id, "buffer_overflow")
    func = _func_from_path(file_entry["file"])
    comp = _component_from_path(file_entry["file"])

    templates = VULN_TEMPLATES.get(vuln_type, {})
    tmpl = templates.get(lang)
    if not tmpl:
        # fallback to any available language template for this vuln type
        for fallback_lang in ["C", "Python", "Go", "JavaScript"]:
            tmpl = templates.get(fallback_lang)
            if tmpl:
                break
    if not tmpl:
        # ultimate fallback
        tmpl = list(list(VULN_TEMPLATES.values())[0].values())[0]

    return tmpl.format(
        func_name=func,
        component=comp,
        obj_type=f"{comp}_obj",
        guard=comp.upper(),
    )


def generate_clean_snippet(file_entry):
    """Generate safe code for a clean file."""
    lang = file_entry.get("file_language", "C")
    func = _func_from_path(file_entry["file"])
    comp = _component_from_path(file_entry["file"])

    templates = CLEAN_TEMPLATES.get(lang, [DEFAULT_CLEAN])
    tmpl = random.choice(templates)

    return tmpl.format(
        func_name=func,
        component=comp,
        guard=comp.upper(),
    )


def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "cve_training_data.json")
    with open(data_path) as f:
        data = json.load(f)

    snippets = {}
    buggy_count = 0
    clean_count = 0

    for entry in data:
        fpath = entry["file"]
        cve_id = entry["cveId"]
        is_buggy = entry["label"] == 1

        if fpath in snippets:
            continue

        if is_buggy:
            code = generate_vuln_snippet(entry, cve_id)
            buggy_count += 1
        else:
            code = generate_clean_snippet(entry)
            clean_count += 1

        snippets[fpath] = code

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "code_snippets.json")
    with open(out_path, "w") as f:
        json.dump(snippets, f, indent=2)

    print(f"Generated {len(snippets)} code snippets")
    print(f"  Buggy: {buggy_count}")
    print(f"  Clean: {clean_count}")

    # Show a sample
    for fpath, code in list(snippets.items())[:2]:
        entry = next(e for e in data if e["file"] == fpath)
        print(f"\n{'='*60}")
        print(f"File: {fpath} ({'BUGGY' if entry['label']==1 else 'CLEAN'})")
        print(f"{'='*60}")
        print(code)


if __name__ == "__main__":
    main()
