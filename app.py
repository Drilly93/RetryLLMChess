"""
Play Chess like a Honey Bee - Chess Challenge Arena

This Gradio app provides:
1. Leaderboard of submitted models
2. Model evaluation interface
3. Submission guide
4. Webhook endpoint for automatic evaluation

The goal is to train a language model to play chess, under a strict constraint:
less than 1M parameters! This is approximately the number of neurons of a honey bee.

Leaderboard data is stored in a private HuggingFace dataset for persistence.
"""

import hashlib
import hmac
import io
import json
import os
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

# Configuration
ORGANIZATION = os.environ.get("HF_ORGANIZATION", "LLM-course")
LEADERBOARD_DATASET = os.environ.get("LEADERBOARD_DATASET", f"{ORGANIZATION}/chess-challenge-leaderboard")
LEADERBOARD_FILENAME = "leaderboard.csv"
HF_TOKEN = os.environ.get("HF_TOKEN")  # Required for private dataset access
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "459f4c2c6b0b4b6468e21f981103753d14219d4955f07ab457e100fee93cae66")

# CSV columns for the leaderboard
LEADERBOARD_COLUMNS = [
    "model_id",
    "user_id",
    "legal_rate",
    "legal_rate_first_try",
    "last_updated",
]


# =============================================================================
# Webhook Queue and Worker
# =============================================================================

eval_queue = queue.Queue()
eval_status = {}  # Track status of queued evaluations
eval_lock = threading.Lock()


def evaluation_worker():
    """Background worker that processes evaluation queue."""
    while True:
        try:
            model_id = eval_queue.get()
            
            with eval_lock:
                eval_status[model_id] = "running"
            
            print(f"[Webhook Worker] Starting evaluation for: {model_id}")
            
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from src.evaluate import (
                    ChessEvaluator,
                    load_model_and_tokenizer,
                    post_discussion_summary,
                )
                
                # Load and evaluate
                model, tokenizer, _ = load_model_and_tokenizer(model_id, verbose=True)
                evaluator = ChessEvaluator(model=model, tokenizer=tokenizer, model_path=model_id)
                result = evaluator.evaluate(verbose=True)
                
                # Update leaderboard if evaluation succeeded
                if result.passed_param_check and result.passed_pychess_check and not result.error_message:
                    user_id = get_model_submitter(model_id)
                    if user_id:
                        leaderboard = load_leaderboard()
                        user_entry = next((e for e in leaderboard if e.get("user_id") == user_id), None)
                        
                        new_entry = {
                            "model_id": model_id,
                            "user_id": user_id,
                            "n_parameters": result.n_parameters,
                            "legal_rate_first_try": result.legal_rate_first_try,
                            "legal_rate_with_retry": result.legal_rate_with_retry,
                            "games_played": result.games_played,
                            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        }
                        
                        if user_entry is None:
                            leaderboard.append(new_entry)
                            save_leaderboard(leaderboard)
                            print(f"[Webhook Worker] Added {model_id} to leaderboard")
                        elif result.legal_rate_with_retry > user_entry.get("legal_rate_with_retry", 0):
                            user_entry.update(new_entry)
                            save_leaderboard(leaderboard)
                            print(f"[Webhook Worker] Updated {model_id} on leaderboard (improvement)")
                        else:
                            print(f"[Webhook Worker] {model_id} - no improvement, not updating leaderboard")
                        
                        # Post results to model discussion
                        if HF_TOKEN:
                            try:
                                post_discussion_summary(model_id, result, HF_TOKEN)
                                print(f"[Webhook Worker] Posted results to {model_id} discussion")
                            except Exception as e:
                                print(f"[Webhook Worker] Failed to post discussion: {e}")
                    else:
                        print(f"[Webhook Worker] Could not determine submitter for {model_id}")
                else:
                    print(f"[Webhook Worker] Evaluation failed for {model_id}: {result.error_message}")
                
                with eval_lock:
                    eval_status[model_id] = "completed"
                    
            except Exception as e:
                print(f"[Webhook Worker] Error evaluating {model_id}: {e}")
                with eval_lock:
                    eval_status[model_id] = f"error: {str(e)}"
                    
        except Exception as e:
            print(f"[Webhook Worker] Queue error: {e}")
        finally:
            eval_queue.task_done()


# Start the background worker thread
worker_thread = threading.Thread(target=evaluation_worker, daemon=True)
worker_thread.start()
print("[Webhook] Evaluation worker started")


def is_chess_model(model_id: str) -> bool:
    """Check if a model ID looks like a chess challenge submission."""
    if not model_id.startswith(f"{ORGANIZATION}/"):
        return False
    model_name = model_id.split("/")[-1].lower()
    return "chess" in model_name


def verify_webhook_signature(body: bytes, signature: str) -> bool:
    """Verify the webhook signature using HMAC-SHA256."""
    if not WEBHOOK_SECRET:
        return True  # Skip verification if no secret configured
    expected = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature or "", expected)


# =============================================================================
# Leaderboard Management
# =============================================================================

def load_leaderboard() -> list:
    """Load leaderboard from private HuggingFace dataset."""
    try:
        from huggingface_hub import hf_hub_download
        
        csv_path = hf_hub_download(
            repo_id=LEADERBOARD_DATASET,
            filename=LEADERBOARD_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        
        df = pd.read_csv(csv_path)
        # Map 'legal_rate' column to 'legal_rate_with_retry' if present
        if 'legal_rate_with_retry' not in df.columns and 'legal_rate' in df.columns:
            df['legal_rate_with_retry'] = df['legal_rate']
        return df.to_dict(orient="records")
    
    except Exception as e:
        print(f"Could not load leaderboard from dataset: {e}")
        return []


def save_leaderboard(data: list):
    """Save leaderboard to private HuggingFace dataset."""
    try:
        from huggingface_hub import HfApi
        
        df = pd.DataFrame(data, columns=LEADERBOARD_COLUMNS)
        
        # Fill missing columns with defaults
        for col in LEADERBOARD_COLUMNS:
            if col not in df.columns:
                df[col] = None
        
        df = df[LEADERBOARD_COLUMNS]
        
        # Convert to CSV bytes
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # Upload to HuggingFace dataset
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=csv_buffer,
            path_in_repo=LEADERBOARD_FILENAME,
            repo_id=LEADERBOARD_DATASET,
            repo_type="dataset",
            commit_message=f"Update leaderboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )
        print(f"Leaderboard saved to {LEADERBOARD_DATASET}")
        
    except Exception as e:
        print(f"Error saving leaderboard to dataset: {e}")
        raise


def get_available_models() -> list:
    """Fetch available models from the organization, newest first, one per user."""
    try:
        from huggingface_hub import list_models
        
        models = list(list_models(author=ORGANIZATION, sort="lastModified", direction=-1))
        chess_models = [m for m in models if "chess" in m.id.lower()]
        
        # Keep only the latest model per user
        seen_users = set()
        filtered_models = []
        for m in chess_models:
            model_name = m.id.split("/")[-1]
            parts = model_name.split("-")
            if len(parts) >= 2:
                username = parts[1] if parts[0] == "chess" else None
                if username and username not in seen_users:
                    seen_users.add(username)
                    filtered_models.append(m.id)
            else:
                filtered_models.append(m.id)
        
        return filtered_models if filtered_models else ["No models available"]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["No models available"]


def get_model_submitter(model_id: str) -> Optional[str]:
    """Extract the submitter's username from the model's README on HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        import re
        
        readme_path = hf_hub_download(
            repo_id=model_id,
            filename="README.md",
            token=HF_TOKEN,
        )
        
        with open(readme_path, "r") as f:
            readme_content = f.read()
        
        match = re.search(r'\*\*Submitted by\*\*:\s*\[([^\]]+)\]', readme_content)
        if match:
            return match.group(1)
        
        from huggingface_hub import model_info
        info = model_info(model_id, token=HF_TOKEN)
        if info.author:
            return info.author
            
    except Exception as e:
        print(f"Could not extract submitter from model: {e}")
    
    return None


# =============================================================================
# Leaderboard Formatting
# =============================================================================

def format_leaderboard_html(data: list) -> str:
    """Format leaderboard data as HTML table."""
    if not data:
        return "<p>No models evaluated yet. Be the first to submit!</p>"
    
    # Keep only the best entry per user (by legal_rate_with_retry)
    best_per_user = {}
    for entry in data:
        user_id = entry.get("user_id", "unknown")
        legal_rate = entry.get("legal_rate_with_retry", 0)
        if user_id not in best_per_user or legal_rate > best_per_user[user_id].get("legal_rate_with_retry", 0):
            best_per_user[user_id] = entry
    
    # Sort by legal_rate_with_retry
    sorted_data = sorted(best_per_user.values(), key=lambda x: x.get("legal_rate_with_retry", 0), reverse=True)
    
    html = """
    <style>
        .leaderboard-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .leaderboard-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
        }
        .leaderboard-table td {
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color-primary, #ddd);
        }
        .rank-1 { color: #ffd700; font-weight: bold; }
        .rank-2 { color: #c0c0c0; font-weight: bold; }
        .rank-3 { color: #cd7f32; font-weight: bold; }
        .model-link { color: #667eea; text-decoration: none; }
        .model-link:hover { text-decoration: underline; }
        .legal-good { color: #28a745; }
        .legal-medium { color: #ffc107; }
        .legal-bad { color: #dc3545; }
    </style>
    <table class="leaderboard-table">
        <thead>
            <tr>
                <th>Rank</th>
                <th>User</th>
                <th>Model</th>
                <th>Legal Rate (with retries)</th>
                <th>Legal Rate (1st try)</th>
                <th>Last Updated</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for i, entry in enumerate(sorted_data, 1):
        rank_class = f"rank-{i}" if i <= 3 else ""
        rank_display = str(i)
        model_url = f"https://huggingface.co/{entry['model_id']}"
        # Color code legal rate
        legal_rate = entry.get('legal_rate_with_retry', 0)
        if legal_rate >= 0.9:
            legal_class = "legal-good"
        elif legal_rate >= 0.7:
            legal_class = "legal-medium"
        else:
            legal_class = "legal-bad"
        user_id = entry.get('user_id', 'unknown')
        user_url = f"https://huggingface.co/{user_id}"
        legal_rate_first = entry.get('legal_rate_first_try', 0)
        html += f"""
            <tr>
                <td class="{rank_class}">{rank_display}</td>
                <td><a href="{user_url}" target="_blank" class="model-link">{user_id}</a></td>
                <td><a href="{model_url}" target="_blank" class="model-link">{entry['model_id'].split('/')[-1]}</a></td>
                <td class="{legal_class}">{legal_rate*100:.1f}%</td>
                <td>{legal_rate_first*100:.1f}%</td>
                <td>{entry.get('last_updated', 'N/A')}</td>
            </tr>
        """
    
    html += "</tbody></table>"
    return html


# =============================================================================
# Evaluation Functions
# =============================================================================

def run_evaluation(
    model_id: str,
    progress: gr.Progress = gr.Progress(),
) -> str:
    """
    Run evaluation on a model and update the leaderboard.
    
    Evaluation procedure:
    1. Check if model has < 1M parameters
    2. Check if model uses python-chess illegally
    3. Play 500 moves against opponent engine (restart after 25 moves)
    4. Track legal move rates
    5. Update leaderboard and post discussion
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        
        from src.evaluate import (
            ChessEvaluator,
            load_model_and_tokenizer,
            post_discussion_summary,
        )
        
        progress(0, desc="Loading model...")
        
        # Load model
        model, tokenizer, _ = load_model_and_tokenizer(model_id, verbose=True)
        
        progress(0.1, desc="Setting up evaluator...")
        
        # Create evaluator
        evaluator = ChessEvaluator(
            model=model,
            tokenizer=tokenizer,
            model_path=model_id,
        )
        
        progress(0.2, desc="Running evaluation (500 moves)...")
        
        # Run evaluation
        result = evaluator.evaluate(verbose=True)
        
        progress(0.9, desc="Updating leaderboard...")
        
        # Check if evaluation was successful
        if not result.passed_param_check:
            return f"""## Evaluation Failed

**Model**: `{model_id}`
**Parameters**: {result.n_parameters:,}

Model exceeds the **1M parameter limit**. Please reduce model size and resubmit.
"""
        
        if not result.passed_pychess_check:
            return f"""## Evaluation Failed

**Model**: `{model_id}`

Model illegally uses python-chess for move filtering: {result.error_message}

This is not allowed. The model must generate moves without access to legal move lists.
"""
        
        if result.error_message:
            return f"""## Evaluation Error

**Model**: `{model_id}`

An error occurred during evaluation: {result.error_message}
"""
        
        # Get submitter info
        user_id = get_model_submitter(model_id)
        if user_id is None:
            return f"""## Evaluation Issue

Could not determine the submitter for model `{model_id}`.

Please ensure your model was submitted using the official submission script (`submit.py`), 
which adds the required metadata to the README.md file.

**Evaluation Results** (not saved to leaderboard):
{result.summary()}
"""
        
        # Update leaderboard
        leaderboard = load_leaderboard()
        
        # Find existing entry for this user
        user_entry = next((e for e in leaderboard if e.get("user_id") == user_id), None)
        
        new_entry = {
            "model_id": model_id,
            "user_id": user_id,
            "n_parameters": result.n_parameters,
            "legal_rate_first_try": result.legal_rate_first_try,
            "legal_rate_with_retry": result.legal_rate_with_retry,
            "games_played": result.games_played,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        
        if user_entry is None:
            leaderboard.append(new_entry)
            save_leaderboard(leaderboard)
            update_message = "New entry added to leaderboard!"
        else:
            old_rate = user_entry.get("legal_rate_with_retry", 0)
            if result.legal_rate_with_retry > old_rate:
                user_entry.update(new_entry)
                save_leaderboard(leaderboard)
                update_message = f"Improved! {old_rate*100:.1f}% -> {result.legal_rate_with_retry*100:.1f}%"
            else:
                update_message = f"No improvement. Best: {old_rate*100:.1f}%, This run: {result.legal_rate_with_retry*100:.1f}%"
        
        # Post discussion to model page
        if HF_TOKEN:
            try:
                post_discussion_summary(model_id, result, HF_TOKEN)
                discussion_message = "Results posted to model page"
            except Exception as e:
                discussion_message = f"Could not post to model page: {e}"
        else:
            discussion_message = "No HF_TOKEN - results not posted to model page"
        
        progress(1.0, desc="Done!")
        
        return f"""## Evaluation Complete

{result.summary()}

---

### Leaderboard Update
{update_message}

### Model Page Discussion
{discussion_message}
"""
        
    except Exception as e:
        import traceback
        return f"""## Evaluation Failed

An unexpected error occurred:

```
{traceback.format_exc()}
```
"""


def refresh_leaderboard() -> str:
    """Refresh and return the leaderboard HTML."""
    return format_leaderboard_html(load_leaderboard())


# =============================================================================
# Gradio Interface
# =============================================================================

with gr.Blocks(
    title="Play Chess like a Honey Bee",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("""
    # 🐝 Play Chess like a Honey Bee
    
    Welcome to the Chess Challenge! The goal is to train a language model to play chess,
    under a strict constraint: **less than 1M parameters!**
    
    This is approximately the number of neurons of a honey bee 🐝
    """)
    
    with gr.Tabs():
        # How to Submit Tab
        with gr.TabItem("📖 How to Submit"):
            gr.Markdown(f"""
            ### Submitting Your Model
                        
            The goal is to create a chess-playing language model with **under 1 million parameters**, 
            which is roughly the number of neurons in a honey bee's brain.
            
            At this scale, efficiency and clever architecture choices are key! We are not targeting 
            superhuman performance, but rather exploring how well small models can learn the rules 
            of chess. The goal is to play **legal moves**.
            
            ---
            
            ### Getting Started
            
            1. **Clone this repository**: 
                ```bash
                git clone https://huggingface.co/spaces/LLM-course/Chess1MChallenge
                ```
                
            2. **Check the example solution** in the `example_solution/` folder for reference
            
            3. **Train your model** using the provided training script or your own approach
            
            4. **Submit using the official script**:
                ```bash
                python submit.py --model_path ./my_model --model_name my-chess-model
                ```
            
            5. **Run evaluation** on this page to see your results on the leaderboard
            
            ---
            
            ### Evaluation Procedure
            
            Your model will be evaluated as follows:
            
            1. **Parameter check**: Must have < 1M parameters
            2. **Security check**: Model cannot use python-chess to filter legal moves
            3. **Game play**: 500 moves against opponent engine (games restart every 25 moves)
            4. **Move generation**: 3 retries allowed per move, greedy decoding
            5. **Scoring**: Legal move rate (first try and with retries)
            
            The evaluation is **fully deterministic** (seeded randomness, deterministic opponent).
            
            ---
            
            ### Requirements
            
            - Model must be under **1M parameters**
            - Model must use the `ChessConfig` and `ChessForCausalLM` classes (or compatible)
            - Include the tokenizer with your submission
            - **Do not** use python-chess to filter moves during generation
            
            ### Tips for Better Performance
            
            - Experiment with different architectures (layers, heads, dimensions)
            - Try weight tying to save parameters
            - Focus on learning the rules of chess, not just memorizing openings
            - Check the `example_solution/` folder for ideas
            """)
        
        # Evaluation Tab
        with gr.TabItem("Evaluate Model"):
            gr.Markdown("""
            ### Run Evaluation
            
            Select a model to evaluate. The evaluation will:
            - Check parameter count (< 1M required)
            - Verify no illegal python-chess usage
            - Play 500 moves against opponent engine
            - Track legal move rates
            - Update the leaderboard (if improvement)
            - Post results to the model's discussion page
            """)
            
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label="Model to Evaluate",
                    scale=4,
                )
                refresh_models_btn = gr.Button("Refresh", scale=1, min_width=50)
            
            def refresh_models():
                return gr.update(choices=get_available_models())
            
            refresh_models_btn.click(
                refresh_models,
                outputs=[model_dropdown],
            )
            
            eval_btn = gr.Button("Run Evaluation", variant="primary")
            eval_results = gr.Markdown()
            
            eval_btn.click(
                run_evaluation,
                inputs=[model_dropdown],
                outputs=eval_results,
            )
        
        # Leaderboard Tab
        with gr.TabItem("Leaderboard"):
            gr.Markdown("### Current Rankings")
            gr.Markdown("""
            Rankings are based on **legal move rate (with retries)**.
            
            - **Legal Rate (1st try)**: Percentage of moves that were legal on first attempt
            - **Legal Rate (with retries)**: Percentage of moves that were legal within 3 attempts
            """)
            
            leaderboard_html = gr.HTML(value=format_leaderboard_html(load_leaderboard()))
            refresh_btn = gr.Button("Refresh Leaderboard")
            refresh_btn.click(refresh_leaderboard, outputs=leaderboard_html)


# =============================================================================
# Webhook Endpoint (mounted on Gradio's FastAPI app)
# =============================================================================

from fastapi import Request
from fastapi.responses import JSONResponse

@demo.app.post("/webhook")
async def handle_webhook(request: Request):
    """
    Handle HuggingFace webhook events for automatic model evaluation.
    
    Triggered on model creation and update events in the organization.
    """
    # Verify webhook signature
    body = await request.body()
    signature = request.headers.get("X-Webhook-Signature")
    
    if not verify_webhook_signature(body, signature):
        print("[Webhook] Invalid signature")
        return JSONResponse({"error": "Invalid signature"}, status_code=401)
    
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    event = payload.get("event", {})
    repo = payload.get("repo", {})
    
    action = event.get("action")
    scope = event.get("scope")
    repo_type = repo.get("type")
    repo_name = repo.get("name", "")
    
    print(f"[Webhook] Received: action={action}, scope={scope}, type={repo_type}, repo={repo_name}")
    
    # Only process model repos in our organization
    if repo_type != "model":
        return JSONResponse({"status": "ignored", "reason": "not a model"})
    
    if not repo_name.startswith(f"{ORGANIZATION}/"):
        return JSONResponse({"status": "ignored", "reason": "not in organization"})
    
    # Only process create and update actions
    if action not in ("create", "update"):
        return JSONResponse({"status": "ignored", "reason": f"action {action} not handled"})
    
    # Check if it looks like a chess model
    if not is_chess_model(repo_name):
        return JSONResponse({"status": "ignored", "reason": "not a chess model"})
    
    # Check if already queued or running
    with eval_lock:
        current_status = eval_status.get(repo_name)
        if current_status == "running":
            return JSONResponse({"status": "ignored", "reason": "evaluation already running"})
        if current_status == "queued":
            return JSONResponse({"status": "ignored", "reason": "already in queue"})
        eval_status[repo_name] = "queued"
    
    # Queue the model for evaluation
    eval_queue.put(repo_name)
    queue_size = eval_queue.qsize()
    
    print(f"[Webhook] Queued {repo_name} for evaluation (queue size: {queue_size})")
    
    return JSONResponse({
        "status": "queued",
        "model_id": repo_name,
        "queue_position": queue_size,
    })


@demo.app.get("/webhook/status")
async def webhook_status():
    """Get the current status of the evaluation queue."""
    with eval_lock:
        status_copy = dict(eval_status)
    
    return JSONResponse({
        "queue_size": eval_queue.qsize(),
        "evaluations": status_copy,
    })


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
