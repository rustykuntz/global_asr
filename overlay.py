import os
import re
import subprocess
import sys


def _arg_value(flag, default=None):
    if flag not in sys.argv:
        return default
    try:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    except Exception:
        pass
    return default


def _parse_duration(default_duration):
    raw = _arg_value("--duration", None)
    if raw is None:
        return default_duration
    try:
        value = float(raw)
        if value <= 0:
            return None
        return value
    except Exception:
        pass
    return default_duration


def _parse_parent_pid():
    raw = _arg_value("--parent-pid", None)
    if raw is None:
        return None
    try:
        pid = int(raw)
        if pid > 0:
            return pid
    except Exception:
        pass
    return None


def _pid_exists(pid):
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


def _parse_payload():
    is_success = "--success" in sys.argv
    custom_text = _arg_value("--text", None)
    custom_color = _arg_value("--color", "green")

    if custom_text:
        label_text = custom_text
    elif is_success:
        label_text = "Sent!"
    else:
        label_text = "Command"

    if custom_color == "red":
        color = "red"
    else:
        color = "green"

    return is_success, label_text, color


def _positive_float(raw):
    try:
        value = float(raw)
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


def _normalized_scale(value):
    return max(0.75, min(3.0, round(value * 4) / 4))


def _linux_overlay_scale(screen_w, screen_h, width_mm=None, height_mm=None):
    override = _positive_float(os.getenv("ASR_OVERLAY_SCALE"))
    if override is not None:
        return _normalized_scale(override)

    gdk_scale = _positive_float(os.getenv("GDK_SCALE"))
    if gdk_scale is not None:
        gdk_dpi_scale = _positive_float(os.getenv("GDK_DPI_SCALE")) or 1.0
        toolkit_scale = gdk_scale * gdk_dpi_scale
        if toolkit_scale > 1.0:
            return _normalized_scale(toolkit_scale)

    qt_scale = _positive_float(os.getenv("QT_SCALE_FACTOR"))
    if qt_scale is not None and qt_scale > 1.0:
        return _normalized_scale(qt_scale)

    xft_dpi = _positive_float(os.getenv("XFT_DPI"))
    if xft_dpi is None:
        try:
            proc = subprocess.run(
                ["xrdb", "-query"],
                capture_output=True,
                text=True,
                check=False,
            )
            match = re.search(r"(?mi)^Xft\.dpi:\s*([0-9.]+)", proc.stdout)
            if match:
                xft_dpi = _positive_float(match.group(1))
        except Exception:
            pass
    if xft_dpi is not None and xft_dpi > 96:
        return _normalized_scale(xft_dpi / 96.0)

    dpi_values = []
    if width_mm and width_mm > 0:
        dpi_values.append(screen_w * 25.4 / width_mm)
    if height_mm and height_mm > 0:
        dpi_values.append(screen_h * 25.4 / height_mm)
    plausible_dpi = [dpi for dpi in dpi_values if 70 <= dpi <= 400]
    if plausible_dpi:
        physical_scale = sum(plausible_dpi) / len(plausible_dpi) / 96.0
        if physical_scale > 1.0:
            return _normalized_scale(physical_scale)

    if screen_w >= 5000 or screen_h >= 2800:
        return 2.0
    if screen_w >= 3200 or screen_h >= 1800:
        return 1.5
    return 1.0


def _parse_xrandr_monitor_geometry(output):
    for line in output.splitlines():
        # Example: " 0: +*DP-1 2560/621x1440/342+0+0  DP-1"
        match = re.search(
            r"\s(\d+)(?:/(\d+))?x(\d+)(?:/(\d+))?([+-]\d+)([+-]\d+)",
            line,
        )
        if not match:
            continue
        width, width_mm, height, height_mm, x, y = match.groups()
        return (
            int(x),
            int(y),
            int(width),
            int(height),
            int(width_mm) if width_mm else None,
            int(height_mm) if height_mm else None,
        )
    return None


def _run_macos_overlay():
    import AppKit
    import objc

    is_success, label_text, color = _parse_payload()
    parent_pid = _parse_parent_pid()

    class OverlayController(AppKit.NSObject):
        def init(self):
            self = objc.super(OverlayController, self).init()
            self.duration = _parse_duration(1.5 if is_success else 2.5)
            self.label_text = label_text
            self.color = color
            self.is_success = is_success
            self.parent_pid = parent_pid
            return self

        def applicationDidFinishLaunching_(self, notification):
            screen = AppKit.NSScreen.mainScreen()
            screen_rect = screen.visibleFrame()

            w, h = 170, 52
            x = screen_rect.origin.x + screen_rect.size.width - w - 20
            y = screen_rect.origin.y + screen_rect.size.height - h - 20
            rect = AppKit.NSMakeRect(x, y, w, h)

            style_mask = AppKit.NSWindowStyleMaskBorderless
            self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                rect, style_mask, AppKit.NSBackingStoreBuffered, False
            )

            self.window.setLevel_(25)
            self.window.setOpaque_(False)
            self.window.setBackgroundColor_(AppKit.NSColor.clearColor())
            self.window.setIgnoresMouseEvents_(True)

            view = OverlayView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, w, h))
            view.label_text = self.label_text
            view.color = self.color
            view.is_success = self.is_success
            self.window.setContentView_(view)

            self.window.makeKeyAndOrderFront_(None)
            self.window.orderFrontRegardless()

            if self.duration is not None:
                AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    self.duration, self, "closeWindow:", None, False
                )
            if self.parent_pid is not None:
                AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    0.7, self, "checkParent:", None, True
                )

        def closeWindow_(self, timer):
            AppKit.NSApplication.sharedApplication().terminate_(self)

        def checkParent_(self, timer):
            if not _pid_exists(self.parent_pid):
                AppKit.NSApplication.sharedApplication().terminate_(self)

    class OverlayView(AppKit.NSView):
        def drawRect_(self, rect):
            bg_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.1, 0.1, 0.82)
            bg_path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, 10, 10)
            bg_color.set()
            bg_path.fill()

            text_color = AppKit.NSColor.whiteColor()
            font = AppKit.NSFont.boldSystemFontOfSize_(14)
            attrs = {
                AppKit.NSForegroundColorAttributeName: text_color,
                AppKit.NSFontAttributeName: font,
            }
            text = AppKit.NSString.stringWithString_(self.label_text)
            text_size = text.sizeWithAttributes_(attrs)
            text_rect = AppKit.NSMakeRect(14, (rect.size.height - text_size.height) / 2, text_size.width, text_size.height)
            text.drawInRect_withAttributes_(text_rect, attrs)

            symbol_color = AppKit.NSColor.redColor() if self.color == "red" else AppKit.NSColor.greenColor()
            if self.is_success:
                path = AppKit.NSBezierPath.bezierPath()
                path.setLineWidth_(2.5)
                path.moveToPoint_(AppKit.NSMakePoint(rect.size.width - 30, rect.size.height / 2))
                path.lineToPoint_(AppKit.NSMakePoint(rect.size.width - 24, rect.size.height / 2 - 5))
                path.lineToPoint_(AppKit.NSMakePoint(rect.size.width - 15, rect.size.height / 2 + 6))
                symbol_color.set()
                path.stroke()
            else:
                dot_rect = AppKit.NSMakeRect(rect.size.width - 25, (rect.size.height - 10) / 2, 10, 10)
                dot = AppKit.NSBezierPath.bezierPathWithOvalInRect_(dot_rect)
                symbol_color.set()
                dot.fill()

    app = AppKit.NSApplication.sharedApplication()
    delegate = OverlayController.alloc().init()
    app.setDelegate_(delegate)
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    app.run()


def _run_windows_overlay():
    import tkinter as tk

    is_success, label_text, color = _parse_payload()
    duration_s = _parse_duration(1.5 if is_success else 2.5)
    parent_pid = _parse_parent_pid()

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)

    width, height = 220, 58
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = max(0, screen_w - width - 24)
    y = max(0, screen_h - height - 60)
    root.geometry(f"{width}x{height}+{x}+{y}")

    bg = "#1a1a1a"
    fg = "#ffffff"
    dot = "#e74c3c" if color == "red" else "#2ecc71"

    frame = tk.Frame(root, bg=bg, bd=0)
    frame.pack(fill="both", expand=True)

    label = tk.Label(frame, text=label_text, bg=bg, fg=fg, font=("Segoe UI", 12, "bold"), anchor="w")
    label.place(x=14, y=16)

    canvas = tk.Canvas(frame, width=14, height=14, bg=bg, highlightthickness=0, bd=0)
    canvas.place(x=width - 28, y=22)
    if is_success:
        canvas.create_line(1, 8, 5, 12, 13, 2, fill=dot, width=2)
    else:
        canvas.create_oval(2, 2, 12, 12, fill=dot, outline=dot)

    if duration_s is not None:
        root.after(int(duration_s * 1000), root.destroy)

    def check_parent():
        if parent_pid is None:
            return
        if not _pid_exists(parent_pid):
            root.destroy()
            return
        root.after(700, check_parent)

    check_parent()
    root.mainloop()


def _run_linux_overlay():
    import ctypes
    import time

    is_success, label_text, color = _parse_payload()
    duration_s = _parse_duration(1.5 if is_success else 2.5)
    parent_pid = _parse_parent_pid()

    if not os.environ.get("DISPLAY"):
        return

    try:
        x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
    except OSError:
        return

    c_int = ctypes.c_int
    c_uint = ctypes.c_uint
    c_ulong = ctypes.c_ulong
    c_long = ctypes.c_long
    c_char_p = ctypes.c_char_p
    c_void_p = ctypes.c_void_p

    class XSetWindowAttributes(ctypes.Structure):
        _fields_ = [
            ("background_pixmap", c_ulong),
            ("background_pixel", c_ulong),
            ("border_pixmap", c_ulong),
            ("border_pixel", c_ulong),
            ("bit_gravity", c_int),
            ("win_gravity", c_int),
            ("backing_store", c_int),
            ("backing_planes", c_ulong),
            ("backing_pixel", c_ulong),
            ("save_under", c_int),
            ("event_mask", c_long),
            ("do_not_propagate_mask", c_long),
            ("override_redirect", c_int),
            ("colormap", c_ulong),
            ("cursor", c_ulong),
        ]

    class XFontStruct(ctypes.Structure):
        _fields_ = [
            ("ext_data", c_void_p),
            ("fid", c_ulong),
        ]

    x11.XOpenDisplay.argtypes = [c_char_p]
    x11.XOpenDisplay.restype = c_void_p
    x11.XDefaultScreen.argtypes = [c_void_p]
    x11.XDefaultScreen.restype = c_int
    x11.XRootWindow.argtypes = [c_void_p, c_int]
    x11.XRootWindow.restype = c_ulong
    x11.XDisplayWidth.argtypes = [c_void_p, c_int]
    x11.XDisplayWidth.restype = c_int
    x11.XDisplayHeight.argtypes = [c_void_p, c_int]
    x11.XDisplayHeight.restype = c_int
    x11.XCreateWindow.argtypes = [
        c_void_p,
        c_ulong,
        c_int,
        c_int,
        c_uint,
        c_uint,
        c_uint,
        c_int,
        c_uint,
        c_void_p,
        c_ulong,
        ctypes.POINTER(XSetWindowAttributes),
    ]
    x11.XCreateWindow.restype = c_ulong
    x11.XCreateGC.argtypes = [c_void_p, c_ulong, c_ulong, c_void_p]
    x11.XCreateGC.restype = c_void_p
    x11.XSetForeground.argtypes = [c_void_p, c_void_p, c_ulong]
    x11.XFillRectangle.argtypes = [c_void_p, c_ulong, c_void_p, c_int, c_int, c_uint, c_uint]
    x11.XDrawString.argtypes = [c_void_p, c_ulong, c_void_p, c_int, c_int, c_char_p, c_int]
    x11.XFillArc.argtypes = [c_void_p, c_ulong, c_void_p, c_int, c_int, c_uint, c_uint, c_int, c_int]
    x11.XDrawLine.argtypes = [c_void_p, c_ulong, c_void_p, c_int, c_int, c_int, c_int]
    x11.XSetLineAttributes.argtypes = [c_void_p, c_void_p, c_uint, c_int, c_int, c_int]
    x11.XLoadQueryFont.argtypes = [c_void_p, c_char_p]
    x11.XLoadQueryFont.restype = ctypes.POINTER(XFontStruct)
    x11.XSetFont.argtypes = [c_void_p, c_void_p, c_ulong]
    x11.XFreeFont.argtypes = [c_void_p, ctypes.POINTER(XFontStruct)]
    x11.XMapRaised.argtypes = [c_void_p, c_ulong]
    x11.XRaiseWindow.argtypes = [c_void_p, c_ulong]
    x11.XStoreName.argtypes = [c_void_p, c_ulong, c_char_p]
    x11.XFlush.argtypes = [c_void_p]
    x11.XDestroyWindow.argtypes = [c_void_p, c_ulong]
    x11.XCloseDisplay.argtypes = [c_void_p]

    display = x11.XOpenDisplay(None)
    if not display:
        return

    def active_monitor_geometry(default_w, default_h):
        try:
            proc = subprocess.run(
                ["xrandr", "--listactivemonitors"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                proc = subprocess.run(
                    ["xrandr", "--listmonitors"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            geometry = _parse_xrandr_monitor_geometry(proc.stdout)
            if geometry:
                return geometry
        except Exception:
            pass
        return 0, 0, default_w, default_h, None, None

    def pixel(hex_color):
        return int(hex_color.lstrip("#"), 16)

    try:
        screen = x11.XDefaultScreen(display)
        root = x11.XRootWindow(display, screen)
        display_w = x11.XDisplayWidth(display, screen)
        display_h = x11.XDisplayHeight(display, screen)
        mon_x, mon_y, screen_w, screen_h, width_mm, height_mm = active_monitor_geometry(
            display_w,
            display_h,
        )
        scale = _linux_overlay_scale(screen_w, screen_h, width_mm, height_mm)

        def scaled(value):
            return max(1, round(value * scale))

        width, height = scaled(220), scaled(58)
        margin = scaled(24)
        x = mon_x + max(0, screen_w - width - margin)
        y = mon_y + margin

        attrs = XSetWindowAttributes()
        attrs.background_pixel = pixel("#1a1a1a")
        attrs.override_redirect = 1

        InputOutput = 1
        CopyFromParent = 0
        CWBackPixel = 1 << 1
        CWOverrideRedirect = 1 << 9
        window = x11.XCreateWindow(
            display,
            root,
            x,
            y,
            width,
            height,
            0,
            CopyFromParent,
            InputOutput,
            None,
            CWBackPixel | CWOverrideRedirect,
            ctypes.byref(attrs),
        )
        gc = x11.XCreateGC(display, window, 0, None)
        x11.XStoreName(display, window, b"global_asr overlay")

        font_px = scaled(14)
        core_font = None
        core_font_names = [
            f"-*-helvetica-bold-r-normal--{font_px}-*-*-*-*-*-iso8859-1",
            f"-misc-fixed-bold-r-normal--{font_px}-*-*-*-*-*-iso8859-1",
        ]
        if font_px >= 23:
            core_font_names.append("12x24")
        elif font_px >= 18:
            core_font_names.append("10x20")
        elif font_px >= 15:
            core_font_names.append("9x15bold")
        else:
            core_font_names.append("8x13bold")
        core_font_names.append("fixed")
        for font_name in core_font_names:
            core_font = x11.XLoadQueryFont(display, font_name.encode("ascii"))
            if core_font:
                x11.XSetFont(display, gc, core_font.contents.fid)
                break

        fg = pixel("#ffffff")
        bg = pixel("#1a1a1a")
        accent = pixel("#e74c3c" if color == "red" else "#2ecc71")
        text_bytes = label_text.replace("●", "").strip().encode("ascii", errors="replace")

        def draw():
            x11.XSetForeground(display, gc, bg)
            x11.XFillRectangle(display, window, gc, 0, 0, width, height)
            x11.XSetForeground(display, gc, fg)
            x11.XDrawString(
                display,
                window,
                gc,
                scaled(14),
                scaled(34),
                text_bytes,
                len(text_bytes),
            )
            x11.XSetForeground(display, gc, accent)
            if is_success:
                x11.XSetLineAttributes(display, gc, scaled(3), 0, 1, 0)
                x11.XDrawLine(
                    display,
                    window,
                    gc,
                    width - scaled(32),
                    scaled(31),
                    width - scaled(27),
                    scaled(38),
                )
                x11.XDrawLine(
                    display,
                    window,
                    gc,
                    width - scaled(27),
                    scaled(38),
                    width - scaled(16),
                    scaled(20),
                )
            else:
                dot_size = scaled(12)
                x11.XFillArc(
                    display,
                    window,
                    gc,
                    width - scaled(28),
                    scaled(22),
                    dot_size,
                    dot_size,
                    0,
                    360 * 64,
                )
            x11.XRaiseWindow(display, window)
            x11.XFlush(display)

        x11.XMapRaised(display, window)
        draw()

        start = time.monotonic()
        while True:
            if duration_s is not None and time.monotonic() - start >= duration_s:
                break
            if parent_pid is not None and not _pid_exists(parent_pid):
                break
            draw()
            time.sleep(0.25)

        if core_font:
            x11.XFreeFont(display, core_font)
        x11.XDestroyWindow(display, window)
        x11.XFlush(display)
    finally:
        x11.XCloseDisplay(display)


def main():
    if sys.platform == "darwin":
        _run_macos_overlay()
        return

    if sys.platform.startswith("win"):
        _run_windows_overlay()
        return

    if sys.platform.startswith("linux"):
        _run_linux_overlay()
        return


if __name__ == "__main__":
    main()
