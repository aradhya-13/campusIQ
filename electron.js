const { app, BrowserWindow, shell } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

const PORT = 8765; // dedicated port so it doesn't clash with the dev server
let pyProc = null;
let mainWindow = null;

// ── Start the FastAPI backend ────────────────────────────────────────────────
function startPython() {
  const projectDir = __dirname;
  pyProc = spawn('python3', ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', String(PORT)], {
    cwd: projectDir,
    env: { ...process.env },
  });

  pyProc.stdout.on('data', d => console.log('[python]', d.toString().trim()));
  pyProc.stderr.on('data', d => console.error('[python]', d.toString().trim()));
  pyProc.on('exit', code => console.log('[python] exited', code));
}

// ── Wait until the server is accepting connections ───────────────────────────
function waitForServer(retries = 40, delay = 500) {
  return new Promise((resolve, reject) => {
    function attempt(n) {
      const req = http.get(`http://127.0.0.1:${PORT}/api/documents`, res => {
        resolve();
      });
      req.on('error', () => {
        if (n <= 0) return reject(new Error('Server did not start in time'));
        setTimeout(() => attempt(n - 1), delay);
      });
      req.end();
    }
    attempt(retries);
  });
}

// ── Create the Electron window ───────────────────────────────────────────────
async function createWindow() {
  startPython();

  // Show a loading splash while waiting
  const splash = new BrowserWindow({
    width: 420, height: 260,
    frame: false, alwaysOnTop: true,
    resizable: false, transparent: true,
    webPreferences: { nodeIntegration: false },
  });
  splash.loadURL(`data:text/html,
    <html><body style="margin:0;display:flex;align-items:center;justify-content:center;
    height:100vh;background:linear-gradient(135deg,#f9c5d1,#c5cff9);font-family:Inter,sans-serif;border-radius:20px">
    <div style="text-align:center">
      <div style="font-size:40px;margin-bottom:12px">📄</div>
      <div style="font-size:20px;font-weight:700;letter-spacing:-.02em">Document Chat</div>
      <div style="font-size:13px;color:#52525b;margin-top:6px">Starting up…</div>
    </div></body></html>`);

  try {
    await waitForServer();
  } catch (e) {
    console.error('Backend failed to start:', e);
  }

  mainWindow = new BrowserWindow({
    width: 1280, height: 820,
    minWidth: 900, minHeight: 600,
    titleBarStyle: 'hiddenInset',  // macOS: clean title bar
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
    show: false,
    icon: path.join(__dirname, 'static', 'icon.png'), // optional icon
  });

  mainWindow.loadURL(`http://127.0.0.1:${PORT}`);

  mainWindow.once('ready-to-show', () => {
    splash.close();
    mainWindow.show();
  });

  // Open external links in the system browser, not Electron
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => { mainWindow = null; });
}

// ── Lifecycle ────────────────────────────────────────────────────────────────
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (pyProc) { pyProc.kill(); pyProc = null; }
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (mainWindow === null) createWindow();
});

app.on('before-quit', () => {
  if (pyProc) { pyProc.kill(); pyProc = null; }
});
