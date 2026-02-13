import AppKit
import Quartz
import sys
import time
import objc


def _parse_duration(default_duration):
    if "--duration" not in sys.argv:
        return default_duration
    try:
        idx = sys.argv.index("--duration")
        if idx + 1 < len(sys.argv):
            value = float(sys.argv[idx + 1])
            if value > 0:
                return value
    except Exception:
        pass
    return default_duration

# Create a borderless, transparent window
class OverlayController(AppKit.NSObject):
    def init(self):
        self = objc.super(OverlayController, self).init()
        self.duration = 4.5 # Default
        return self

    def applicationDidFinishLaunching_(self, notification):
        # Screen Size
        screen = AppKit.NSScreen.mainScreen()
        screen_rect = screen.visibleFrame() # Use visibleFrame to respect Menu Bar/Dock
        
        # Window Size
        w, h = 140, 50
        
        # Position: Top Right
        # visibleFrame origin is usually (0, DockHeight) or similar.
        # Max Y is screen_rect.origin.y + screen_rect.size.height
        
        x = screen_rect.origin.x + screen_rect.size.width - w - 20
        y = screen_rect.origin.y + screen_rect.size.height - h - 20 
        
        rect = AppKit.NSMakeRect(x, y, w, h)
        
        # Window Style
        style_mask = AppKit.NSWindowStyleMaskBorderless
        
        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style_mask, AppKit.NSBackingStoreBuffered, False
        )
        
        # Levels: kCGStatusWindowLevel (25) > kCGFloatingWindowLevel (5)
        self.window.setLevel_(25) 
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(AppKit.NSColor.clearColor())
        self.window.setIgnoresMouseEvents_(True) 
        
        # View
        view = OverlayView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, w, h))
        self.window.setContentView_(view)
        
        self.window.makeKeyAndOrderFront_(None)
        self.window.orderFrontRegardless() # Force show even if app is background
        
        # Close Timer
        AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            self.duration, self, "closeWindow:", None, False
        )

    def closeWindow_(self, timer):
        AppKit.NSApplication.sharedApplication().terminate_(self)

class OverlayView(AppKit.NSView):
    def drawRect_(self, rect):
        is_success = "--success" in sys.argv
        
        # Parse custom text and color
        custom_text = None
        custom_color = "green"
        
        if "--text" in sys.argv:
            try:
                idx = sys.argv.index("--text")
                if idx + 1 < len(sys.argv):
                    custom_text = sys.argv[idx + 1]
            except:
                pass
        
        if "--color" in sys.argv:
            try:
                idx = sys.argv.index("--color")
                if idx + 1 < len(sys.argv):
                    custom_color = sys.argv[idx + 1]
            except:
                pass

        # Draw Background (Rounded Rectangle with slight Alpha)
        bg_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.1, 0.1, 0.8)
        bg_path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, 10, 10)
        bg_color.set()
        bg_path.fill()
        
        # Config
        if custom_text:
            label_text = custom_text
            if custom_color == "red":
                 symbol_color = AppKit.NSColor.redColor()
            else:
                 symbol_color = AppKit.NSColor.greenColor()
        elif is_success:
            label_text = "Sent!"
            symbol_color = AppKit.NSColor.greenColor()
        else:
            label_text = "Command"
            symbol_color = AppKit.NSColor.greenColor()

        # Draw Text
        text_color = AppKit.NSColor.whiteColor()
        font = AppKit.NSFont.boldSystemFontOfSize_(14)
        attrs = {
            AppKit.NSForegroundColorAttributeName: text_color,
            AppKit.NSFontAttributeName: font
        }
        text = AppKit.NSString.stringWithString_(label_text)
        text_size = text.sizeWithAttributes_(attrs)
        
        text_rect = AppKit.NSMakeRect(
            15, (rect.size.height - text_size.height) / 2, 
            text_size.width, text_size.height
        )
        text.drawInRect_withAttributes_(text_rect, attrs)
        
        # Draw Symbol (Dot or Checkmark)
        symbol_rect = AppKit.NSMakeRect(rect.size.width - 30, (rect.size.height - 15) / 2, 15, 15)
        
        if is_success:
            # Draw Checkmark
            path = AppKit.NSBezierPath.bezierPath()
            path.setLineWidth_(2.5)
            # Simple tick shape
            path.moveToPoint_(AppKit.NSMakePoint(symbol_rect.origin.x, symbol_rect.origin.y + 8))
            path.lineToPoint_(AppKit.NSMakePoint(symbol_rect.origin.x + 5, symbol_rect.origin.y + 3))
            path.lineToPoint_(AppKit.NSMakePoint(symbol_rect.origin.x + 12, symbol_rect.origin.y + 12))
            symbol_color.set()
            path.stroke()
        else:
            # Draw Dot
            dot_path = AppKit.NSBezierPath.bezierPathWithOvalInRect_(AppKit.NSMakeRect(rect.size.width - 25, (rect.size.height - 10) / 2, 10, 10))
            symbol_color.set()
            dot_path.fill()

def main():
    app = AppKit.NSApplication.sharedApplication()
    delegate = OverlayController.alloc().init()
    app.setDelegate_(delegate)
    
    # Duration logic
    duration = 1.5 if "--success" in sys.argv else 2.5
    delegate.duration = _parse_duration(duration)
    
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    app.run()

if __name__ == "__main__":
    main()
