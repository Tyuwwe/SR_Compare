# Simulate real OS-level mouse input (SetForegroundWindow + SetCursorPos +
# PostMessage WM_MOUSEWHEEL) against the sr_compare GUI window, to prove the
# interactive wheel-zoom path: wndproc -> ImGui win32 backend -> io.MouseWheel
# -> handleCompareZoomInput.  Waits for the window to appear first.
import ctypes, sys, time
from ctypes import wintypes

user32 = ctypes.windll.user32
WM_MOUSEWHEEL = 0x020A
WHEEL_DELTA = 120

EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

def find_window():
    found = []
    def enum_cb(hwnd, lparam):
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if buf.value.startswith("sr_compare"):
                found.append(hwnd)
        return True
    user32.EnumWindows(EnumWindowsProc(enum_cb), 0)
    return found[0] if found else 0

hwnd = 0
for _ in range(120):
    hwnd = find_window()
    if hwnd:
        break
    time.sleep(0.5)
if not hwnd:
    print("window not found")
    sys.exit(1)
print(f"hwnd={hwnd}")

# Foreground + cursor over the render area (right of the 360px panel) so the
# backend's focused-window mouse-pos fallback feeds io.MousePos.
user32.SetForegroundWindow(hwnd)
time.sleep(0.5)
pt = wintypes.POINT(960, 540)
user32.ClientToScreen(hwnd, ctypes.byref(pt))
user32.SetCursorPos(pt.x, pt.y)
time.sleep(0.2)
# SetCursorPos does not generate WM_MOUSEMOVE; post it explicitly (client
# coords) so the backend feeds io.MousePos even without foreground focus.
move_lparam = (540 << 16) | (960 & 0xFFFF)
user32.PostMessageW(hwnd, 0x0200, 0, move_lparam)  # WM_MOUSEMOVE
time.sleep(0.5)

lparam = (pt.y << 16) | (pt.x & 0xFFFF)
for i in range(6):
    user32.PostMessageW(hwnd, WM_MOUSEWHEEL, WHEEL_DELTA << 16, lparam)
    time.sleep(0.2)
    print(f"posted wheel-up #{i + 1} t={time.strftime('%H:%M:%S')}")
print("done")
