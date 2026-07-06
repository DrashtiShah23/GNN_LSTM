# Remote Dashboard From Local Computer

This keeps the Streamlit dashboard running on this computer and exposes it through a temporary public tunnel.

Do not port-forward your router for this dashboard. Use a tunnel so you can stop sharing by closing the tunnel process.

## 1. Start the Dashboard

From the repo root:

```powershell
.\scripts\start_dashboard_local.ps1 -Port 8501
```

Leave that PowerShell window open.

If Streamlit is not installed in the venv:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit
```

## 2. Expose It With a Tunnel

Open a second PowerShell window.

### Option A: Cloudflare Quick Tunnel

Install once:

```powershell
winget install Cloudflare.cloudflared
```

Run:

```powershell
cloudflared tunnel --url http://localhost:8501
```

Copy the generated `https://*.trycloudflare.com` URL and share it.

### Option B: ngrok

Install once:

```powershell
winget install Ngrok.Ngrok
```

After ngrok account setup/auth:

```powershell
ngrok http 8501
```

Copy the generated `https://*.ngrok-free.app` URL and share it.

## Notes

- The dashboard is read-only with respect to experiment results, but it exposes result artifacts and predictions to anyone with the URL.
- Close the tunnel window to revoke access.
- Close the Streamlit window to stop the dashboard.
- Keep the laptop awake and connected to the internet while people are viewing it.
- The default dashboard result-set filter opens on `canonical_protocol_only` when present.
