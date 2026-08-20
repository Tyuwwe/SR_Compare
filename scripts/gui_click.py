# Click the "GT 200% SSAA" checkbox with a real OS-level left click.
import ctypes, sys, time
from ctypes import wintypes

user32 = ctypes.windll.user32

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

hwnd = find_window()
if not hwnd:
    print("window not found")
    sys.exit(1)

# Checkbox client coords (measured from a screen capture): ~(60, 141).
pt = wintypes.POINT(60, 141)
user32.ClientToScreen(hwnd, ctypes.byref(pt))
print(f"click at screen ({pt.x},{pt.y})")
user32.SetCursorPos(pt.x, pt.y)
time.sleep(0.3)
user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
time.sleep(0.1)
user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
print("clicked")
