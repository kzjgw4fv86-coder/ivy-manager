# Ivy Manager

Custom control panel for **Ivy**, the AI receptionist for **Alfie Alan Music**.

Works on your iPhone (and any browser) via Streamlit Cloud.

---

## Deploy to Streamlit Cloud (recommended)

### 1. Create a GitHub repository
- Go to [github.com](https://github.com) and create a new **public** repository (e.g. `ivy-manager`).
- Upload all the files from this folder (`app.py`, `requirements.txt`, `ivy_prompt.txt`, `.streamlit/`, etc.).

### 2. Deploy on Streamlit
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select your repository, branch `main`, and set the main file path to `app.py`.
4. Click **Deploy**.

### 3. Add your API key
1. Once the app is live, click the **⋮** menu (top right) → **Settings** → **Secrets**.
2. Paste exactly this and save:

```toml
XAI_API_KEY = "your_actual_xai_key_here"
```

3. The app will restart. Ivy is now ready.

### 4. Use on iPhone
- Open the Streamlit app link in **Safari**.
- Tap **Share** → **Add to Home Screen**.
- It now opens like a normal app.

---

## Important note about data

On the free Streamlit Cloud plan the app sleeps when unused and the local SQLite database is wiped on restart.

- Use it for testing and managing calls day-to-day.
- If a call is important, open **Call Detail** and copy the details somewhere safe.

---

## Files

- `app.py` — the full application
- `requirements.txt` — dependencies
- `ivy_prompt.txt` — original Ivy prompt (also editable inside the app)
- `.streamlit/config.toml` — dark theme + settings

Built for Alfie Alan Music.
