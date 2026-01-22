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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

# Configuration
ORGANIZATION = os.environ.get("HF_ORGANIZATION", "LLM-course")
LEADERBOARD_DATASET = os.environ.get("LEADERBOARD_DATASET", f"{ORGANIZATION}/chess-challenge-leaderboard")
LEADERBOARD_FILENAME = "leaderboard.csv"
HF_TOKEN = os.environ.get("HF_TOKEN")  # Required for private dataset access

# CSV columns for the leaderboard
LEADERBOARD_COLUMNS = [
    "model_id",
    "user_id",
    "legal_rate",
    "legal_rate_first_try",
    "last_updated",
    "model_last_modified",
]

def is_chess_model(model_id: str) -> bool:
    """Check if a model ID looks like a chess challenge submission."""
    if not model_id.startswith(f"{ORGANIZATION}/"):
        return False
    model_name = model_id.split("/")[-1].lower()
    return "chess" in model_name

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
    
    # Keep only the best entry per user (by legal_rate)
    best_per_user = {}
    for entry in data:
        user_id = entry.get("user_id", "unknown")
        legal_rate = entry.get("legal_rate", 0)
        if user_id not in best_per_user or legal_rate > best_per_user[user_id].get("legal_rate", 0):
            best_per_user[user_id] = entry
    
    # Sort by legal_rate
    sorted_data = sorted(best_per_user.values(), key=lambda x: x.get("legal_rate", 0), reverse=True)
    
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
        legal_rate = entry.get('legal_rate', 0)
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

        from src.evaluate import (
            ChessEvaluator,
            load_model_and_tokenizer,
            post_discussion_summary,
        )
        from huggingface_hub import model_info as hf_model_info

        progress(0, desc="Getting model info...")
        try:
            model_info = hf_model_info(model_id, token=HF_TOKEN)
            model_last_modified = model_info.lastModified
        except Exception as e:
            return f"## Evaluation Failed \
Could not fetch model info for `{model_id}`: {e}"

        leaderboard = load_leaderboard()
        model_entry = next((e for e in leaderboard if e.get("model_id") == model_id), None)

        if model_entry and "last_updated" in model_entry and model_entry["last_updated"]:
            last_evaluation_date = datetime.strptime(model_entry["last_updated"], "%Y-%m-%d %H:%M")
            
            # model_last_modified is timezone-aware, last_evaluation_date is naive.
            # Compare them by making model_last_modified naive UTC.
            if last_evaluation_date > model_last_modified.astimezone(timezone.utc).replace(tzinfo=None):
                return f"""## Evaluation Skipped

Model `{model_id}` was already evaluated on {last_evaluation_date.strftime('%Y-%m-%d %H:%M UTC')}
which is after the model was last modified on {model_last_modified.strftime('%Y-%m-%d %H:%M UTC')}.

No new evaluation is needed.
"""
        
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

        print("=" * 80)
        print(f"Evaluation summary for {model_id}")
        print(result.summary())
        print("=" * 80)
        
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
        
        # Find existing entry for this model
        model_entry = next((e for e in leaderboard if e.get("model_id") == model_id), None)
        
        new_entry = {
            "model_id": model_id,
            "user_id": user_id,
            "legal_rate": result.legal_rate_with_retry,
            "legal_rate_first_try": result.legal_rate_first_try,
            "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "model_last_modified": model_last_modified.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        }
        
        if model_entry is None:
            leaderboard.append(new_entry)
            save_leaderboard(leaderboard)
            update_message = "New entry added to leaderboard!"
        else:
            old_rate = model_entry.get("legal_rate", 0)
            model_entry.update(new_entry) # Update existing entry for the model
            save_leaderboard(leaderboard)
            if result.legal_rate_with_retry > old_rate:
                update_message = f"Improved! {old_rate*100:.1f}% -> {result.legal_rate_with_retry*100:.1f}%"
            else:
                update_message = f"Re-evaluated. Previous: {old_rate*100:.1f}%, This run: {result.legal_rate_with_retry*100:.1f}%"
        update_message = f"No improvement. Best: {old_rate*100:.1f}%, This run: {result.legal_rate*100:.1f}%"
        
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
                git clone ssh://huggingface.co/spaces/LLM-course/Chess1MChallenge
                ```
                
            2. **Check an example solution** in the `example_solution/` folder for reference
            
            3. **Train your model** using the provided training script or your own approach, and evaluate it locally:
                ```bash
                python -m src --model ./my_model
                ```
            
            
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
                    allow_custom_value=True,
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


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
