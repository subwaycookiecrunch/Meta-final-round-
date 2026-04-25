"""
CodeReviewEnv v3 — HuggingFace Space App (RECOVERY VERSION)
=========================================================
This version includes:
1. Auto-refreshing logs (via gr.Timer)
2. Persistent log saving (to survive Space restarts)
3. Auto-boot training
"""
import gradio as gr
import os
import sys
import json
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

# ── Paths ──────────────────────────────────────────────
RESULTS_DIR = "./grpo_output"
LOG_FILE = os.path.join(RESULTS_DIR, "live_training_logs.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── State Persistence ──────────────────────────────────
def load_logs():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return f.read()
        except:
            return "Error reading logs."
    return "⏸️ Ready to Train. Waiting for boot..."

def save_logs(text):
    try:
        with open(LOG_FILE, "w") as f:
            f.write(text)
    except:
        pass

TRAINING_DONE = os.path.exists(os.path.join(RESULTS_DIR, "training_stats.json"))
training_status = {
    "running": False, 
    "progress": load_logs(), 
    "done": TRAINING_DONE
}

# ── Background Process ─────────────────────────────────
def run_training():
    """Run GRPO training in a background thread."""
    training_status["running"] = True
    training_status["done"] = False
    training_status["progress"] = "🚀 Initializing training script..."
    save_logs(training_status["progress"])
    
    try:
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, "train_grpo.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(__file__) or "."
        )
        lines = []
        for line in proc.stdout:
            lines.append(line.strip())
            # Keep last 30 lines for visibility
            training_status["progress"] = "\n".join(lines[-30:])
            save_logs(training_status["progress"])
            print(line.strip()) # Also to container logs

        exit_code = proc.wait()
        if exit_code == 0 and os.path.exists(os.path.join(RESULTS_DIR, "training_stats.json")):
            training_status["done"] = True
            training_status["progress"] = "✅ Training Complete!"
        else:
            training_status["done"] = False
            error_tail = "\n".join(lines[-20:])
            training_status["progress"] = f"❌ Training FAILED (Exit {exit_code}).\n\nRecent output:\n{error_tail}"
        
        save_logs(training_status["progress"])
        training_status["running"] = False
    except Exception as e:
        training_status["running"] = False
        training_status["done"] = False
        training_status["progress"] = f"❌ CRITICAL ERROR: {str(e)}"
        save_logs(training_status["progress"])

def start_training_btn():
    if training_status["running"]:
        return "⚠️ Already running!"
    training_status["done"] = False
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    return "🚀 Manually started. Watch the logs below."

# ── UI Layout ──────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="CodeReviewEnv v3") as app:
    gr.Markdown("# 🔐 CodeReviewEnv v3 — Agentic Security Dashboard")
    
    with gr.Tabs():
        with gr.Tab("🏋️ Training Progress"):
            status_header = gr.Markdown("### Initializing...")
            
            with gr.Row():
                manual_btn = gr.Button("🚀 Force Start", variant="secondary", size="sm")
                refresh_btn = gr.Button("🔄 Manual Refresh", size="sm")
            
            # The Timer: refreshes the UI every 2 seconds automatically
            timer = gr.Timer(2)
            
            with gr.Group():
                gr.Markdown("#### Live Training Output (Auto-refreshing)")
                output_text = gr.Code(label=None, lines=20, interactive=False)
            
            with gr.Group():
                gr.Markdown("#### Results Visualization")
                plot_img = gr.Image(label=None, value=os.path.join(RESULTS_DIR, "training_curves.png") if os.path.exists(os.path.join(RESULTS_DIR, "training_curves.png")) else None)

            def update_ui():
                # Check for completion
                if not training_status["done"] and os.path.exists(os.path.join(RESULTS_DIR, "training_stats.json")):
                    training_status["done"] = True
                
                # Check for crash
                if not training_status["running"] and not training_status["done"]:
                    header = "### ⏸️ Ready to Train / Crashed"
                elif training_status["done"]:
                    header = "### ✅ Training Complete!"
                else:
                    header = "### ⏳ Training in Progress..."
                
                log_val = training_status["progress"]
                if not log_val or log_val == "Initializing...":
                    log_val = load_logs()
                
                plot_val = os.path.join(RESULTS_DIR, "training_curves.png") if os.path.exists(os.path.join(RESULTS_DIR, "training_curves.png")) else None
                
                return header, log_val, plot_val

            # Auto-updates
            timer.tick(update_ui, outputs=[status_header, output_text, plot_img])
            refresh_btn.click(update_ui, outputs=[status_header, output_text, plot_img])
            manual_btn.click(start_training_btn, outputs=[output_text])

        with gr.Tab("🔍 Demo & About"):
            gr.Markdown("### Coming soon...")
            gr.Markdown("The model is currently training. This tab will activate once results are ready.")

if __name__ == "__main__":
    # AUTO-BOOT: Start training immediately on launch
    if not training_status["done"] and not training_status["running"]:
        print("🚀 [BOOT] Starting background training thread...")
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
    
    app.launch(server_name="0.0.0.0", server_port=7860)
