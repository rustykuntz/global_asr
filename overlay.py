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
        if value > 0:
            return value
    except Exception:
        pass
    return default_duration


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


def _run_macos_overlay():
    import AppKit
    import objc

    is_success, label_text, color = _parse_payload()

    class OverlayController(AppKit.NSObject):
        def init(self):
            self = objc.super(OverlayController, self).init()
            self.duration = _parse_duration(1.5 if is_success else 2.5)
            self.label_text = label_text
            self.color = color
            self.is_success = is_success
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

            AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                self.duration, self, "closeWindow:", None, False
            )

        def closeWindow_(self, timer):
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
    if duration_s <= 0:
        duration_s = 1.5

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

    root.after(int(duration_s * 1000), root.destroy)
    root.mainloop()


def main():
    if sys.platform == "darwin":
        _run_macos_overlay()
        return

    if sys.platform.startswith("win"):
        _run_windows_overlay()
        return


if __name__ == "__main__":
    main()
