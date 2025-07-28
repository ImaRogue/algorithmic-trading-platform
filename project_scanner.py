#!/usr/bin/env python3
"""
ENHANCED NYX PROJECT SCANNER
Comprehensive analysis of current project state + database logging diagnosis
"""

import ast
import sqlite3
from pathlib import Path
from datetime import datetime
import json


class EnhancedNYXScanner:
    """Advanced project scanner with database logging analysis"""

    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        self.analysis = {}
        self.database_status = {}
        self.critical_issues = []

    def run_complete_scan(self):
        """Run comprehensive project analysis"""
        print("🚨 ENHANCED NYX PROJECT SCANNER")
        print("=" * 50)
        print(f"📁 Scanning: {self.project_dir.absolute()}")
        print(f"⏰ Scan time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Phase 1: File Discovery
        self.discover_files()

        # Phase 2: Code Analysis
        self.analyze_code_structure()

        # Phase 3: Database Analysis
        self.analyze_database_status()

        # Phase 4: Trading System Analysis
        self.analyze_trading_system()

        # Phase 5: Critical Issues
        self.identify_critical_issues()

        # Phase 6: Actionable Report
        self.generate_action_plan()

        return self.analysis

    def discover_files(self):
        """Discover and categorize all project files"""
        print("🔍 PHASE 1: FILE DISCOVERY")
        print("-" * 30)

        if not self.project_dir.exists():
            print(f"❌ Directory not found: {self.project_dir}")
            return

        all_files = list(self.project_dir.rglob('*'))
        files = [f for f in all_files if f.is_file()]

        # Categorize files
        categories = {
            'python': [],
            'config': [],
            'data': [],
            'logs': [],
            'other': []
        }

        for file_path in files:
            relative_path = file_path.relative_to(self.project_dir)
            size_kb = file_path.stat().st_size / 1024

            if file_path.suffix == '.py':
                categories['python'].append((relative_path, size_kb))
            elif file_path.suffix in ['.env', '.yaml', '.yml', '.json', '.txt', '.cfg']:
                categories['config'].append((relative_path, size_kb))
            elif file_path.suffix in ['.db', '.sqlite', '.csv', '.log']:
                categories['data'].append((relative_path, size_kb))
            elif file_path.name.endswith('.log') or 'log' in str(file_path).lower():
                categories['logs'].append((relative_path, size_kb))
            else:
                categories['other'].append((relative_path, size_kb))

        # Display results
        total_size = sum(f[1] for cat in categories.values() for f in cat)
        print(f"📊 Total files: {len(files)} ({total_size:.1f} KB)")
        print()

        for category, file_list in categories.items():
            if file_list:
                cat_size = sum(f[1] for f in file_list)
                print(f"📁 {category.upper()}: {len(file_list)} files ({cat_size:.1f} KB)")
                for file_path, size in sorted(file_list, key=lambda x: x[1], reverse=True):
                    print(f"   📄 {file_path} ({size:.1f} KB)")
                print()

        self.analysis['file_discovery'] = categories
        return categories

    def analyze_code_structure(self):
        """Analyze Python code structure and capabilities"""
        print("🐍 PHASE 2: CODE STRUCTURE ANALYSIS")
        print("-" * 30)

        python_files = self.analysis.get('file_discovery', {}).get('python', [])
        code_analysis = {}

        for file_path, size in python_files:
            full_path = self.project_dir / file_path
            analysis = self.analyze_python_file(full_path)
            code_analysis[str(file_path)] = analysis

            print(f"🔍 {file_path} ({size:.1f} KB)")
            if analysis['classes']:
                print(f"   📦 Classes: {', '.join(analysis['classes'][:5])}")
            if analysis['key_functions']:
                print(f"   🔧 Key Functions: {', '.join(analysis['key_functions'][:5])}")
            if analysis['capabilities']:
                print(f"   🎯 Capabilities: {', '.join(analysis['capabilities'])}")
            if analysis['trading_methods']:
                print(f"   💰 Trading Methods: {', '.join(analysis['trading_methods'])}")
            if analysis['database_usage']:
                print(f"   🗄️ Database Usage: {analysis['database_usage']}")
            print()

        self.analysis['code_structure'] = code_analysis
        return code_analysis

    def analyze_python_file(self, file_path):
        """Deep analysis of Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            try:
                tree = ast.parse(content)
            except SyntaxError:
                return {'error': 'Syntax error in file', 'size_kb': file_path.stat().st_size / 1024}

            analysis = {
                'size_kb': file_path.stat().st_size / 1024,
                'lines': len(content.split('\n')),
                'classes': [],
                'key_functions': [],
                'all_functions': [],
                'imports': [],
                'capabilities': [],
                'trading_methods': [],
                'database_usage': 'None',
                'has_gui': False,
                'has_trading_logic': False,
                'has_database_calls': False
            }

            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis['classes'].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    analysis['all_functions'].append(node.name)
                    # Identify key trading functions
                    if any(keyword in node.name.lower() for keyword in
                           ['trade', 'execute', 'buy', 'sell', 'order', 'signal']):
                        analysis['trading_methods'].append(node.name)
                        analysis['key_functions'].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)

            # Detect capabilities
            analysis['capabilities'] = self.detect_advanced_capabilities(content)
            analysis['has_gui'] = any(keyword in content for keyword in ['tkinter', 'Tk()', 'mainloop'])
            analysis['has_trading_logic'] = any(keyword in content.lower() for keyword in
                                                ['execute_trade', 'buy', 'sell', 'strategy', 'signal'])
            analysis['has_database_calls'] = any(keyword in content for keyword in
                                                 ['sqlite3', 'database', 'log_trade', 'cursor', 'commit'])

            # Database usage analysis
            if 'sqlite3' in content:
                if 'log_trade' in content or 'INSERT' in content:
                    analysis['database_usage'] = 'Active_Logging'
                elif 'connect' in content:
                    analysis['database_usage'] = 'Connection_Only'
                else:
                    analysis['database_usage'] = 'Import_Only'
            elif any(keyword in content for keyword in ['database', 'db.', 'log_trade']):
                analysis['database_usage'] = 'External_DB_Calls'

            return analysis

        except Exception as e:
            return {'error': str(e), 'size_kb': file_path.stat().st_size / 1024, 'lines': 0}

    def detect_advanced_capabilities(self, content):
        """Detect advanced system capabilities"""
        capabilities = []

        capability_patterns = {
            'GUI_Interface': ['tkinter', 'Tk()', 'mainloop'],
            'Real_Time_Data': ['requests', 'api', 'get_quote', 'real_time'],
            'Trading_Execution': ['execute_trade', 'buy', 'sell', 'place_order'],
            'Strategy_Engine': ['strategy', 'signal', 'crossover', 'rsi'],
            'Risk_Management': ['risk', 'stop_loss', 'position_size', 'max_loss'],
            'Portfolio_Management': ['portfolio', 'balance', 'positions'],
            'Database_Logging': ['sqlite3', 'log_trade', 'database', 'INSERT'],
            'Performance_Tracking': ['performance', 'profit', 'pnl', 'sharpe'],
            'Machine_Learning': ['sklearn', 'tensorflow', 'learning', 'predict'],
            'API_Integration': ['alpha_vantage', 'finnhub', 'polygon', 'api_key'],
            'Visualization': ['matplotlib', 'plot', 'chart', 'graph'],
            'Configuration': ['config', 'yaml', 'json', 'settings']
        }

        for capability, patterns in capability_patterns.items():
            if any(pattern in content.lower() for pattern in patterns):
                capabilities.append(capability)

        return capabilities

    def analyze_database_status(self):
        """Comprehensive database analysis"""
        print("🗄️ PHASE 3: DATABASE STATUS ANALYSIS")
        print("-" * 30)

        database_files = []

        # Find database files
        for file_path, _ in self.analysis.get('file_discovery', {}).get('data', []):
            if file_path.suffix in ['.db', '.sqlite']:
                database_files.append(self.project_dir / file_path)

        if not database_files:
            print("❌ NO DATABASE FILES FOUND")
            print("🚨 CRITICAL: This explains the 592-trade data loss!")
            self.critical_issues.append("NO_DATABASE_FILES")
            self.database_status = {'status': 'MISSING', 'files': [], 'tables': []}
            return

        # Analyze each database
        for db_file in database_files:
            print(f"🔍 Analyzing: {db_file.name}")
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()

                # Get tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                db_analysis = {
                    'file': db_file.name,
                    'size_kb': db_file.stat().st_size / 1024,
                    'tables': [table[0] for table in tables],
                    'trade_count': 0,
                    'last_trade': None
                }

                # Check for trades table
                if any('trade' in table.lower() for table in db_analysis['tables']):
                    try:
                        cursor.execute("SELECT COUNT(*) FROM trades;")
                        db_analysis['trade_count'] = cursor.fetchone()[0]

                        cursor.execute("SELECT MAX(timestamp) FROM trades;")
                        last_trade = cursor.fetchone()[0]
                        db_analysis['last_trade'] = last_trade
                    except:
                        pass

                conn.close()

                print(f"   📊 Size: {db_analysis['size_kb']:.1f} KB")
                print(f"   📋 Tables: {', '.join(db_analysis['tables'])}")
                print(f"   💰 Trade Records: {db_analysis['trade_count']}")
                if db_analysis['last_trade']:
                    print(f"   ⏰ Last Trade: {db_analysis['last_trade']}")

                # Critical check for 592-trade loss
                if db_analysis['trade_count'] == 0 and db_analysis['tables']:
                    print("   🚨 CRITICAL: Tables exist but NO TRADES LOGGED!")
                    self.critical_issues.append("EMPTY_DATABASE_TABLES")

                self.database_status[db_file.name] = db_analysis

            except Exception as e:
                print(f"   ❌ Error analyzing {db_file.name}: {e}")

        print()

    def analyze_trading_system(self):
        """Analyze trading system components"""
        print("💰 PHASE 4: TRADING SYSTEM ANALYSIS")
        print("-" * 30)

        trading_analysis = {
            'main_trading_files': [],
            'has_gui_trading': False,
            'has_strategy_engine': False,
            'has_risk_management': False,
            'has_database_integration': False,
            'trading_methods_found': []
        }

        code_structure = self.analysis.get('code_structure', {})

        for file_path, analysis in code_structure.items():
            if analysis.get('has_trading_logic'):
                trading_analysis['main_trading_files'].append(file_path)

            if analysis.get('has_gui') and analysis.get('has_trading_logic'):
                trading_analysis['has_gui_trading'] = True

            if 'Strategy_Engine' in analysis.get('capabilities', []):
                trading_analysis['has_strategy_engine'] = True

            if 'Risk_Management' in analysis.get('capabilities', []):
                trading_analysis['has_risk_management'] = True

            if analysis.get('has_database_calls'):
                trading_analysis['has_database_integration'] = True

            trading_analysis['trading_methods_found'].extend(analysis.get('trading_methods', []))

        print(f"📄 Trading Files: {', '.join(trading_analysis['main_trading_files'])}")
        print(f"🖥️ GUI Trading: {'✅' if trading_analysis['has_gui_trading'] else '❌'}")
        print(f"🧠 Strategy Engine: {'✅' if trading_analysis['has_strategy_engine'] else '❌'}")
        print(f"🛡️ Risk Management: {'✅' if trading_analysis['has_risk_management'] else '❌'}")
        print(f"🗄️ Database Integration: {'✅' if trading_analysis['has_database_integration'] else '❌'}")
        print(f"⚙️ Trading Methods: {', '.join(set(trading_analysis['trading_methods_found']))}")

        self.analysis['trading_system'] = trading_analysis
        print()

    def identify_critical_issues(self):
        """Identify critical issues preventing profitable trading"""
        print("🚨 PHASE 5: CRITICAL ISSUES ANALYSIS")
        print("-" * 30)

        issues = []

        # Database logging issue
        trading_system = self.analysis.get('trading_system', {})
        database_status = self.database_status

        if trading_system.get('has_gui_trading') and not trading_system.get('has_database_integration'):
            issues.append({
                'severity': 'CRITICAL',
                'issue': 'DATABASE_DISCONNECTION',
                'description': 'Trading system exists but no database logging integration',
                'impact': 'ALL TRADE DATA LOST (592-trade scenario)',
                'fix': 'Add database logging calls to trading execution methods'
            })

        if database_status and any(
                db.get('trade_count', 0) == 0 for db in database_status.values() if isinstance(db, dict)):
            issues.append({
                'severity': 'CRITICAL',
                'issue': 'EMPTY_DATABASE',
                'description': 'Database tables exist but contain no trade records',
                'impact': 'No historical data for analysis or optimization',
                'fix': 'Integrate log_trade() calls in execution flow'
            })

        if not trading_system.get('main_trading_files'):
            issues.append({
                'severity': 'HIGH',
                'issue': 'NO_TRADING_SYSTEM',
                'description': 'No trading execution system found',
                'impact': 'Cannot execute trades',
                'fix': 'Implement trading execution system'
            })

        # Display issues
        for issue in issues:
            print(f"🚨 {issue['severity']}: {issue['issue']}")
            print(f"   📝 {issue['description']}")
            print(f"   💥 Impact: {issue['impact']}")
            print(f"   🔧 Fix: {issue['fix']}")
            print()

        self.critical_issues.extend(issues)
        return issues

    def generate_action_plan(self):
        """Generate specific action plan for immediate fixes"""
        print("🎯 PHASE 6: IMMEDIATE ACTION PLAN")
        print("-" * 30)

        trading_files = self.analysis.get('trading_system', {}).get('main_trading_files', [])

        if not trading_files:
            print("❌ No trading files found - cannot generate specific fix plan")
            return

        main_file = trading_files[0]  # Assume first is main file

        print(f"🎯 PRIMARY TARGET FILE: {main_file}")
        print()
        print("📋 IMMEDIATE FIXES REQUIRED:")
        print()

        # Database integration fix
        print("1. 🗄️ DATABASE LOGGING INTEGRATION")
        print(f"   File to modify: {main_file}")
        print("   Required changes:")
        print("   - Add: from database_logging_fix import TradingDatabase")
        print("   - Add: self.db = TradingDatabase() in __init__")
        print("   - Add: trade_id = self.db.log_trade(...) after each trade")
        print("   - Add: verification check after each log")
        print()

        # Find specific methods that need fixing
        code_analysis = self.analysis.get('code_structure', {}).get(main_file, {})
        trading_methods = code_analysis.get('trading_methods', [])

        if trading_methods:
            print("2. 🔧 SPECIFIC METHODS TO MODIFY:")
            for method in trading_methods:
                print(f"   - {method}(): Add database logging call")
            print()

        print("3. ⚡ IMMEDIATE TESTING:")
        print("   - Create database_logging_fix.py")
        print("   - Modify main trading file")
        print("   - Run verification test")
        print("   - Test with 5-10 trades")
        print("   - Verify data persistence")
        print()

        print("🚀 ESTIMATED TIME TO FIX: 30-60 minutes")
        print("🎯 RESULT: 100% trade capture rate, zero data loss")

    def save_analysis_report(self):
        """Save complete analysis to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_dir / f"project_analysis_{timestamp}.json"

        # Prepare serializable data
        serializable_analysis = {}
        for key, value in self.analysis.items():
            if isinstance(value, dict):
                serializable_analysis[key] = value
            else:
                serializable_analysis[key] = str(value)

        report_data = {
            'timestamp': timestamp,
            'project_dir': str(self.project_dir),
            'analysis': serializable_analysis,
            'database_status': self.database_status,
            'critical_issues': self.critical_issues
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"📄 Analysis report saved: {report_file.name}")


def main():
    """Run the enhanced scanner"""
    scanner = EnhancedNYXScanner()
    analysis = scanner.run_complete_scan()
    scanner.save_analysis_report()

    print("\n" + "=" * 50)
    print("🎯 SCANNER COMPLETE!")
    print("📊 Use this analysis to fix the 592-trade data loss issue!")

    return scanner


if __name__ == "__main__":
    scanner = main()