---
title: Agent Cache Bill Viewer
emoji: 🧾
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: "6.31.0"
app_file: app.py
pinned: false
license: mit
short_description: Replay agent runs and see what prompt caching saves
---

# Agent Cache Bill Viewer

Replay real tool-using agent runs (Qwen3-4B on vLLM with prefix caching) and see, request by
request, which tokens were cached, which were re-read at full price, and what each context policy
really cost.

Code, data and paper: https://github.com/VatsalRoy/keep-it-warm