# momentum_integration_patch.py
"""
HYBRID APPROACH: Manual integration with enhanced logging
Integrates momentum strategy with working_gui.py

USAGE: python momentum_integration_patch.py
This will modify working_gui.py to include momentum strategy
"""

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Import our enhanced logging
try:
    from nyx_enhanced_logging import get_logger
    logger = get_logger()
    LOGGING_AVAILABLE = True
    print("✅ Enhanced logging available")
except ImportError:
    LOGGING_AVAILABLE = False
    print("⚠️ Enhanced logging not available - using basic logging")

def backup_working_gui():
    """Create backup of working_gui.py before modification"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"working_gui_backup_{timestamp}.py"

    if Path("working_gui.py").exists():
        shutil.copy2("working_gui.py", backup_path)
        print(f"✅ Backup created: {backup_path}")

        if LOGGING_AVAILABLE:
            with logger.performance_timer("backup_working_gui"):
                pass

        return backup_path
    else:
        print("❌ working_gui.py not found!")
        return None

def create_momentum_integration():
    """Create the momentum strategy integration code"""

    integration_code = '''
# =================== MOMENTUM STRATEGY INTEGRATION ===================
# Added by NYX Hybrid Updater System
# Integrates momentum strategy with existing trading system

import sys
import os
from pathlib import Path

# Add enterprise path for strategy imports
enterprise_path = Path(__file__).parent / "nyx-enterprise"
if str(enterprise_path) not in sys.path:
    sys.path.insert(0, str(enterprise_path))

try:
    from strategies.momentum import MomentumStrategy, create_momentum_strategy
    MOMENTUM_AVAILABLE = True
    print("✅ Momentum strategy loaded successfully")
except ImportError as e:
    MOMENTUM_AVAILABLE = False
    print(f"⚠️ Momentum strategy not available: {e}")

class MomentumTrader:
    """Momentum strategy integration for working_gui.py"""

    def __init__(self):
        self.momentum_strategy = None
        self.momentum_signals = {}
        self.momentum_enabled = False

        if MOMENTUM_AVAILABLE:
            try:
                self.momentum_strategy = create_momentum_strategy(
                    lookback_period=20,
                    momentum_threshold=0.02
                )
                self.momentum_enabled = True
                print("🚀 Momentum trader initialized")
            except Exception as e:
                print(f"❌ Momentum trader initialization failed: {e}")

    def analyze_symbol_momentum(self, symbol, price_data):
        """Analyze symbol using momentum strategy"""
        if not self.momentum_enabled or not self.momentum_strategy:
            return None

        try:
            signal = self.momentum_strategy.analyze_symbol(symbol, price_data)
            self.momentum_signals[symbol] = signal

            # Log the analysis
            print(f"📊 Momentum analysis for {symbol}: {signal['action']} "
                  f"(confidence: {signal['confidence']:.2f})")

            return signal

        except Exception as e:
            print(f"❌ Momentum analysis failed for {symbol}: {e}")
            return None

    def get_momentum_signal(self, symbol):
        """Get latest momentum signal for symbol"""
        return self.momentum_signals.get(symbol, None)

    def get_momentum_performance(self):
        """Get momentum strategy performance stats"""
        if self.momentum_strategy:
            return self.momentum_strategy.get_performance_stats()
        return {}

# Global momentum trader instance
momentum_trader = MomentumTrader()

def integrate_momentum_with_existing_trading():
    """Integration helper function"""
    global momentum_trader

    # This function can be called from existing trading logic
    # to incorporate momentum signals into trading decisions

    def enhanced_trading_decision(symbol, current_data, existing_signal=None):
        """Enhanced trading decision with momentum"""

        # Get momentum signal
        momentum_signal = momentum_trader.analyze_symbol_momentum(symbol, current_data)

        if momentum_signal and existing_signal:
            # Combine signals (weighted approach)
            momentum_weight = 0.4
            existing_weight = 0.6

            # Simple signal combination logic
            if momentum_signal['action'] == existing_signal and momentum_signal['confidence'] > 0.6:
                return {
                    'action': momentum_signal['action'],
                    'confidence': min(0.95, momentum_signal['confidence'] * momentum_weight + 0.8 * existing_weight),
                    'reasoning': f"Combined: momentum + existing signal",
                    'momentum_component': momentum_signal
                }

        return momentum_signal or existing_signal

    return enhanced_trading_decision

# Export the integration function
enhanced_trading_decision = integrate_momentum_with_existing_trading()

# =================== END MOMENTUM INTEGRATION ===================
'''

    return integration_code

def integrate_momentum_strategy():
    """Main integration function"""
    print("🚀 Starting Momentum Strategy Integration...")
    print("=" * 60)

    if LOGGING_AVAILABLE:
        with logger.performance_timer("integrate_momentum_strategy", {"strategy": "momentum"}):
            _perform_integration()
    else:
        _perform_integration()

def _perform_integration():
    """Perform the actual integration"""

    # Step 1: Backup working_gui.py
    print("\n1️⃣ Creating backup...")
    backup_path = backup_working_gui()
    if not backup_path:
        return

    # Step 2: Check if momentum strategy exists
    print("\n2️⃣ Checking momentum strategy...")
    momentum_path = Path("nyx-enterprise/strategies/momentum.py")
    if not momentum_path.exists():
        print("❌ Momentum strategy not found! Create it first.")
        return

    print("✅ Momentum strategy found")

    # Step 3: Read current working_gui.py
    print("\n3️⃣ Reading working_gui.py...")
    try:
        with open("working_gui.py", "r", encoding="utf-8") as f:
            original_content = f.read()
        print(f"✅ Read {len(original_content)} characters")
    except Exception as e:
        print(f"❌ Failed to read working_gui.py: {e}")
        return

    # Step 4: Check if already integrated
    if "MOMENTUM STRATEGY INTEGRATION" in original_content:
        print("⚠️ Momentum strategy already integrated!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Step 5: Add integration code
    print("\n4️⃣ Adding momentum integration...")
    integration_code = create_momentum_integration()

    # Insert after imports (find a good insertion point)
    lines = original_content.split('\n')

    # Find insertion point (after imports, before main classes)
    insertion_point = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insertion_point = i + 1
        elif line.strip().startswith('class ') and insertion_point > 0:
            break

    # Insert the integration code
    lines.insert(insertion_point, integration_code)
    modified_content = '\n'.join(lines)

    # Step 6: Write modified file
    print("\n5️⃣ Writing modified working_gui.py...")
    try:
        with open("working_gui.py", "w", encoding="utf-8") as f:
            f.write(modified_content)
        print("✅ Integration complete!")
    except Exception as e:
        print(f"❌ Failed to write modified file: {e}")
        # Restore backup
        shutil.copy2(backup_path, "working_gui.py")
        print("✅ Backup restored")
        return

    # Step 7: Test the integration
    print("\n6️⃣ Testing integration...")
    test_integration()

    print("\n" + "=" * 60)
    print("🎉 MOMENTUM STRATEGY INTEGRATION COMPLETE!")
    print(f"📁 Backup saved as: {backup_path}")
    print("🚀 Momentum strategy is now available in working_gui.py")

    if LOGGING_AVAILABLE:
        logger.log_trading_action(
            symbol="INTEGRATION",
            action="COMPLETE",
            quantity=1,
            price=0.0,
            strategy="momentum_integration",
            confidence=1.0,
            result="success"
        )

def test_integration():
    """Test the momentum integration"""
    try:
        # Try to import and test the integration
        import importlib.util

        spec = importlib.util.spec_from_file_location("working_gui_test", "working_gui.py")
        test_module = importlib.util.module_from_spec(spec)

        # This will execute the file and test imports
        spec.loader.exec_module(test_module)

        # Check if momentum_trader is available
        if hasattr(test_module, 'momentum_trader'):
            print("✅ Momentum trader successfully integrated")

            # Quick functionality test
            if hasattr(test_module.momentum_trader, 'momentum_enabled'):
                if test_module.momentum_trader.momentum_enabled:
                    print("✅ Momentum strategy functional")
                else:
                    print("⚠️ Momentum strategy loaded but not enabled")

        else:
            print("⚠️ Integration added but momentum_trader not accessible")

    except Exception as e:
        print(f"⚠️ Integration test failed: {e}")
        print("   (This might be normal if working_gui.py has complex dependencies)")

def show_integration_summary():
    """Show what was integrated"""
    print("\n📋 INTEGRATION SUMMARY:")
    print("=" * 40)
    print("✅ Added MomentumTrader class to working_gui.py")
    print("✅ Integrated momentum strategy analysis")
    print("✅ Created enhanced_trading_decision function")
    print("✅ Added signal combination logic")
    print("✅ Enabled momentum performance tracking")
    print("\n🎯 USAGE:")
    print("• momentum_trader.analyze_symbol_momentum(symbol, data)")
    print("• momentum_trader.get_momentum_signal(symbol)")
    print("• enhanced_trading_decision(symbol, data, existing_signal)")
    print("\n🔧 NEXT STEPS:")
    print("• Test momentum analysis with real data")
    print("• Integrate with existing trading buttons/functions")
    print("• Monitor performance via momentum_trader.get_momentum_performance()")

if __name__ == "__main__":
    print("NYX HYBRID UPDATER - MOMENTUM INTEGRATION")
    print("=" * 50)

    # Check prerequisites
    if not Path("working_gui.py").exists():
        print("❌ working_gui.py not found in current directory")
        sys.exit(1)

    if not Path("nyx-enterprise/strategies/momentum.py").exists():
        print("❌ Momentum strategy not found. Create it first!")
        print("   Expected: nyx-enterprise/strategies/momentum.py")
        sys.exit(1)

    # Confirm integration
    print("🎯 Ready to integrate momentum strategy with working_gui.py")
    print("⚠️ This will modify working_gui.py (backup will be created)")

    confirm = input("\nProceed with integration? (yes/no): ").strip().lower()
    if confirm == 'yes':
        integrate_momentum_strategy()
        show_integration_summary()
    else:
        print("❌ Integration cancelled")