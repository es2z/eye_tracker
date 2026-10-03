-- 視線オーバーレイをOBSと連動して起動するスクリプト
--
-- 使い方: OBSの「ツール」→「スクリプト」でこのファイルを追加する。
-- OBSの起動時に Python(gaze-overlay)をコンソール窓なしで起動する。
-- Python側は --watch-pid でOBSのプロセスを監視し、OBSが終了(異常終了を含む)したら自分も終了する。
-- スクリプトを再読み込みしても、Python側の多重起動防止により二重には起動しない。

obs = obslua
local ffi = require("ffi")

ffi.cdef [[
typedef uint32_t DWORD;
typedef int BOOL;
typedef void* HANDLE;
typedef uint16_t WORD;
typedef struct {
  DWORD cb; void* lpReserved; void* lpDesktop; void* lpTitle;
  DWORD dwX; DWORD dwY; DWORD dwXSize; DWORD dwYSize;
  DWORD dwXCountChars; DWORD dwYCountChars; DWORD dwFillAttribute; DWORD dwFlags;
  WORD wShowWindow; WORD cbReserved2; void* lpReserved2;
  HANDLE hStdInput; HANDLE hStdOutput; HANDLE hStdError;
} STARTUPINFOW;
typedef struct { HANDLE hProcess; HANDLE hThread; DWORD dwProcessId; DWORD dwThreadId; } PROCESS_INFORMATION;
BOOL CreateProcessW(const uint16_t* app, uint16_t* cmd, void* pa, void* ta, BOOL inherit, DWORD flags,
                    void* env, const uint16_t* cwd, STARTUPINFOW* si, PROCESS_INFORMATION* pi);
BOOL CloseHandle(HANDLE h);
DWORD GetCurrentProcessId(void);
DWORD GetLastError(void);
int MultiByteToWideChar(unsigned int cp, DWORD flags, const char* str, int cb, uint16_t* wstr, int cch);
]]

local CP_UTF8 = 65001
local CREATE_NO_WINDOW = 0x08000000

local settings = nil

local function log(msg)
  obs.script_log(obs.LOG_INFO, "[gaze-overlay] " .. msg)
end

-- UTF-8の文字列をUTF-16(Windows APIのW版が受け取る形式)に変換する
local function to_wide(s)
  local n = ffi.C.MultiByteToWideChar(CP_UTF8, 0, s, #s, nil, 0)
  local buf = ffi.new("uint16_t[?]", n + 1)
  ffi.C.MultiByteToWideChar(CP_UTF8, 0, s, #s, buf, n)
  buf[n] = 0
  return buf
end

local function file_exists(path)
  local f = io.open(path, "rb")
  if f then
    f:close()
    return true
  end
  return false
end

-- このスクリプトの1つ上のフォルダをプロジェクトのフォルダとみなす(obs/ の親)
local function default_project_dir()
  local dir = script_path():gsub("[/\\]+$", "")
  return (dir:gsub("[/\\][^/\\]+$", ""))
end

local function project_dir()
  local dir = settings and obs.obs_data_get_string(settings, "project_dir") or ""
  if dir == "" then
    dir = default_project_dir()
  end
  return (dir:gsub("/", "\\"))
end

local function uv_path()
  local p = settings and obs.obs_data_get_string(settings, "uv_path") or ""
  if p ~= "" then
    return (p:gsub("/", "\\"))
  end
  -- uvの標準のインストール先。OBSを起動したときのPATHに含まれていない場合に備える
  local home = os.getenv("USERPROFILE") or ""
  local candidate = home .. "\\.local\\bin\\uv.exe"
  if file_exists(candidate) then
    return candidate
  end
  return "uv"
end

-- コンソール窓を出さずにプロセスを起動する
local function spawn(cmdline, cwd)
  local si = ffi.new("STARTUPINFOW")
  si.cb = ffi.sizeof(si)
  local pi = ffi.new("PROCESS_INFORMATION")
  local wcmd = to_wide(cmdline)
  local wcwd = to_wide(cwd)
  local ok = ffi.C.CreateProcessW(nil, wcmd, nil, nil, 0, CREATE_NO_WINDOW, nil, wcwd, si, pi)
  if ok == 0 then
    log("起動に失敗しました (エラー " .. tonumber(ffi.C.GetLastError()) .. "): " .. cmdline)
    return false
  end
  ffi.C.CloseHandle(pi.hThread)
  ffi.C.CloseHandle(pi.hProcess)
  return true
end

local function base_command()
  local dir = project_dir()
  return string.format('"%s" run --project "%s" gaze-overlay', uv_path(), dir), dir
end

local function start_overlay()
  local cmd, dir = base_command()
  local extra = settings and obs.obs_data_get_string(settings, "extra_args") or ""
  cmd = cmd .. " --watch-pid " .. tonumber(ffi.C.GetCurrentProcessId())
  if extra ~= "" then
    cmd = cmd .. " " .. extra
  end
  if spawn(cmd, dir) then
    log("起動しました: " .. cmd)
  end
end

local function stop_overlay()
  local cmd, dir = base_command()
  if spawn(cmd .. " --stop", dir) then
    log("停止を要求しました")
  end
end

local function restart_tick()
  obs.remove_current_callback()
  start_overlay()
end

local function restart_overlay()
  stop_overlay()
  -- 前のプロセスが終了してから起動する(すぐ起動すると多重起動防止で終了してしまうため)
  obs.timer_add(restart_tick, 3000)
end

function script_description()
  return "視線オーバーレイ(gaze-overlay)をOBSと連動して起動・終了します。\n"
    .. "ブラウザソース: http://127.0.0.1:8765/overlay\n"
    .. "ドック: http://127.0.0.1:8765/dock\n"
    .. "ログ: プロジェクトの data\\logs\\gaze_overlay.log"
end

function script_properties()
  local p = obs.obs_properties_create()
  obs.obs_properties_add_path(p, "project_dir", "プロジェクトのフォルダ(空ならこのスクリプトの親)", obs.OBS_PATH_DIRECTORY, nil, nil)
  obs.obs_properties_add_path(p, "uv_path", "uv.exe(空なら自動)", obs.OBS_PATH_FILE, "uv.exe (uv.exe)", nil)
  obs.obs_properties_add_text(p, "extra_args", "追加の引数(例: --mode monitor)", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_bool(p, "autostart", "OBSの起動時に自動で起動する")
  obs.obs_properties_add_button(p, "start", "起動", function()
    start_overlay()
    return false
  end)
  obs.obs_properties_add_button(p, "stop", "停止", function()
    stop_overlay()
    return false
  end)
  obs.obs_properties_add_button(p, "restart", "再起動", function()
    restart_overlay()
    return false
  end)
  return p
end

function script_defaults(s)
  obs.obs_data_set_default_bool(s, "autostart", true)
end

function script_update(s)
  settings = s
end

function script_load(s)
  settings = s
  if obs.obs_data_get_bool(s, "autostart") then
    start_overlay()
  end
end

-- OBSの終了時に明示的に止めたい場合は、次の関数のコメントを外す。
-- (通常はPython側が --watch-pid でOBSの終了を検知するので不要。スクリプトの再読み込みでも止まるようになる点に注意)
-- function script_unload()
--   stop_overlay()
-- end
