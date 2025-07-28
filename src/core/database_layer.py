# 🗄️ Database Layer - Persistent Storage System
# Advanced database operations with backup, optimization, and integrity checks

import sqlite3
import pandas as pd
import numpy as np
import json
import os
import shutil
import threading
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging
import zipfile
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    db_path: str = "trading_data.db"
    backup_path: str = "backups/"
    max_connections: int = 10
    timeout: int = 30
    wal_mode: bool = True
    cache_size: int = 100  # MB
    backup_interval_hours: int = 24
    cleanup_days: int = 365
    compression: bool = True


class DatabaseManager:
    """
    Advanced database management system for trading platform
    Features: Automatic backups, integrity checks, performance optimization
    """

    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(__name__)

        # Connection pool
        self.connection_pool = []
        self.pool_lock = threading.Lock()

        # Background tasks
        self.background_running = False
        self.backup_thread = None

        # Performance tracking
        self.query_stats = {}
        self.stats_lock = threading.Lock()

        # Initialize database
        self._initialize_database()
        self._start_background_tasks()

    def _initialize_database(self):
        """Initialize database with optimized settings"""

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.config.db_path) or '.', exist_ok=True)
        os.makedirs(self.config.backup_path, exist_ok=True)

        with self.get_connection() as conn:
            # Enable WAL mode for better concurrency
            if self.config.wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")

            # Optimize cache settings
            conn.execute(f"PRAGMA cache_size=-{self.config.cache_size * 1024}")  # Convert MB to KB

            # Other optimizations
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping

            # Create tables
            self._create_tables(conn)

            # Create indexes
            self._create_indexes(conn)

        self.logger.info(f"Database initialized: {self.config.db_path}")

    def _create_tables(self, conn: sqlite3.Connection):
        """Create all required tables"""

        # Market data with partitioning by date
        conn.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                adj_close REAL,
                source TEXT,
                data_quality INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, source)
            )
        ''')

        # Strategy signals and performance
        conn.execute('''
            CREATE TABLE IF NOT EXISTS strategy_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                signal_type TEXT NOT NULL,  -- BUY, SELL, HOLD
                confidence REAL,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                metadata TEXT,  -- JSON for additional signal data
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Trade execution records
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                strategy_name TEXT,
                action TEXT NOT NULL,  -- BUY, SELL
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                commission REAL DEFAULT 0,
                slippage REAL DEFAULT 0,
                timestamp DATETIME NOT NULL,
                order_type TEXT,  -- MARKET, LIMIT, STOP
                status TEXT DEFAULT 'FILLED',  -- PENDING, FILLED, CANCELLED
                pnl REAL,
                metadata TEXT,  -- JSON for additional trade data
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Portfolio snapshots
        conn.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                total_value REAL NOT NULL,
                cash_balance REAL NOT NULL,
                invested_value REAL NOT NULL,
                positions TEXT NOT NULL,  -- JSON of positions
                daily_pnl REAL,
                total_pnl REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Strategy performance metrics
        conn.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                symbol TEXT,
                date DATE NOT NULL,
                total_signals INTEGER DEFAULT 0,
                profitable_signals INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                avg_holding_period REAL,  -- in hours
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strategy_name, symbol, date)
            )
        ''')

        # Risk metrics and alerts
        conn.execute('''
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,  -- POSITION_LIMIT, DAILY_LOSS, etc.
                severity TEXT NOT NULL,    -- LOW, MEDIUM, HIGH, CRITICAL
                symbol TEXT,
                description TEXT NOT NULL,
                value REAL,
                threshold REAL,
                timestamp DATETIME NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # News and sentiment data
        conn.execute('''
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                headline TEXT NOT NULL,
                summary TEXT,
                sentiment_score REAL,  -- -1 to 1
                confidence REAL,       -- 0 to 1
                source TEXT,
                url TEXT,
                publish_time DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # System configuration history
        conn.execute('''
            CREATE TABLE IF NOT EXISTS config_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_type TEXT NOT NULL,  -- STRATEGY, RISK, API, etc.
                config_name TEXT,
                old_value TEXT,
                new_value TEXT,
                changed_by TEXT DEFAULT 'system',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Database metadata and integrity
        conn.execute('''
            CREATE TABLE IF NOT EXISTS db_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Initialize metadata
        conn.execute('''
            INSERT OR REPLACE INTO db_metadata (key, value) 
            VALUES ('schema_version', '1.0')
        ''')
        conn.execute('''
            INSERT OR REPLACE INTO db_metadata (key, value) 
            VALUES ('created_at', ?)
        ''', (datetime.now().isoformat(),))

        conn.commit()

    def _create_indexes(self, conn: sqlite3.Connection):
        """Create optimized indexes for query performance"""

        indexes = [
            # Market data indexes
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_market_data_source ON market_data(source, timestamp DESC)",

            # Strategy signals indexes
            "CREATE INDEX IF NOT EXISTS idx_signals_strategy_time ON strategy_signals(strategy_name, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON strategy_signals(symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_signals_type_time ON strategy_signals(signal_type, timestamp DESC)",

            # Trades indexes
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_trades_strategy_time ON trades(strategy_name, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status, timestamp DESC)",

            # Portfolio snapshots indexes
            "CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_snapshots(timestamp DESC)",

            # Strategy performance indexes
            "CREATE INDEX IF NOT EXISTS idx_strategy_perf_name_date ON strategy_performance(strategy_name, date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_strategy_perf_symbol_date ON strategy_performance(symbol, date DESC)",

            # Risk events indexes
            "CREATE INDEX IF NOT EXISTS idx_risk_events_type_time ON risk_events(event_type, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_risk_events_severity ON risk_events(severity, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_risk_events_resolved ON risk_events(resolved, timestamp DESC)",

            # News sentiment indexes
            "CREATE INDEX IF NOT EXISTS idx_news_symbol_time ON news_sentiment(symbol, publish_time DESC)",
            "CREATE INDEX IF NOT EXISTS idx_news_sentiment_score ON news_sentiment(sentiment_score, publish_time DESC)",

            # Config history indexes
            "CREATE INDEX IF NOT EXISTS idx_config_history_type_time ON config_history(config_type, timestamp DESC)"
        ]

        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except sqlite3.Error as e:
                self.logger.warning(f"Index creation warning: {e}")

        conn.commit()
        self.logger.info("Database indexes created")

    @contextmanager
    def get_connection(self):
        """Get database connection with connection pooling"""
        conn = None
        try:
            # Try to get connection from pool
            with self.pool_lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()

            # Create new connection if pool is empty
            if conn is None:
                conn = sqlite3.connect(
                    self.config.db_path,
                    timeout=self.config.timeout,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row  # Enable column access by name

            yield conn

        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                # Return connection to pool if under limit
                with self.pool_lock:
                    if len(self.connection_pool) < self.config.max_connections:
                        self.connection_pool.append(conn)
                    else:
                        conn.close()

    def execute_query(self, query: str, params: Tuple = None, fetch: str = None) -> Any:
        """
        Execute query with performance tracking

        Args:
            query: SQL query string
            params: Query parameters
            fetch: 'one', 'all', or None
        """
        start_time = time.time()

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                if fetch == 'one':
                    result = cursor.fetchone()
                elif fetch == 'all':
                    result = cursor.fetchall()
                else:
                    result = cursor.rowcount

                conn.commit()

                # Track query performance
                execution_time = time.time() - start_time
                self._track_query_performance(query, execution_time)

                return result

        except sqlite3.Error as e:
            self.logger.error(f"Database query error: {e}")
            self.logger.error(f"Query: {query}")
            raise e

    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Execute many queries efficiently"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount

    def bulk_insert_market_data(self, data_points: List[Dict]):
        """Efficiently insert multiple market data points"""
        if not data_points:
            return 0

        query = '''
            INSERT OR REPLACE INTO market_data 
            (symbol, timestamp, open_price, high_price, low_price, close_price, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''

        params_list = [
            (
                point['symbol'],
                point['timestamp'],
                point.get('open_price'),
                point.get('high_price'),
                point.get('low_price'),
                point.get('close_price'),
                point.get('volume'),
                point.get('source', 'unknown')
            )
            for point in data_points
        ]

        return self.execute_many(query, params_list)

    def get_market_data(self, symbol: str = None, start_date: datetime = None,
                        end_date: datetime = None, limit: int = None) -> pd.DataFrame:
        """Retrieve market data with flexible filtering"""

        query = '''
            SELECT symbol, timestamp, open_price, high_price, low_price, 
                   close_price, volume, source
            FROM market_data
        '''

        conditions = []
        params = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')

        return df

    def record_trade(self, trade_data: Dict) -> str:
        """Record a trade execution"""

        trade_id = trade_data.get(
            'trade_id') or f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(trade_data)) % 10000:04d}"

        query = '''
            INSERT INTO trades 
            (trade_id, symbol, strategy_name, action, quantity, price, commission, 
             slippage, timestamp, order_type, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        params = (
            trade_id,
            trade_data['symbol'],
            trade_data.get('strategy_name'),
            trade_data['action'],
            trade_data['quantity'],
            trade_data['price'],
            trade_data.get('commission', 0),
            trade_data.get('slippage', 0),
            trade_data.get('timestamp', datetime.now()),
            trade_data.get('order_type', 'MARKET'),
            trade_data.get('status', 'FILLED'),
            json.dumps(trade_data.get('metadata', {}))
        )

        self.execute_query(query, params)
        return trade_id

    def update_trade_pnl(self, trade_id: str, pnl: float):
        """Update trade P&L after closing position"""
        query = "UPDATE trades SET pnl = ? WHERE trade_id = ?"
        self.execute_query(query, (pnl, trade_id))

    def record_strategy_signal(self, signal_data: Dict):
        """Record strategy signal"""

        query = '''
            INSERT INTO strategy_signals 
            (strategy_name, symbol, timestamp, signal_type, confidence, 
             entry_price, target_price, stop_loss, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        params = (
            signal_data['strategy_name'],
            signal_data['symbol'],
            signal_data.get('timestamp', datetime.now()),
            signal_data['signal_type'],
            signal_data.get('confidence'),
            signal_data.get('entry_price'),
            signal_data.get('target_price'),
            signal_data.get('stop_loss'),
            json.dumps(signal_data.get('metadata', {}))
        )

        self.execute_query(query, params)

    def record_portfolio_snapshot(self, snapshot_data: Dict):
        """Record portfolio snapshot"""

        query = '''
            INSERT INTO portfolio_snapshots 
            (timestamp, total_value, cash_balance, invested_value, positions, 
             daily_pnl, total_pnl, sharpe_ratio, max_drawdown, win_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        params = (
            snapshot_data.get('timestamp', datetime.now()),
            snapshot_data['total_value'],
            snapshot_data['cash_balance'],
            snapshot_data['invested_value'],
            json.dumps(snapshot_data['positions']),
            snapshot_data.get('daily_pnl'),
            snapshot_data.get('total_pnl'),
            snapshot_data.get('sharpe_ratio'),
            snapshot_data.get('max_drawdown'),
            snapshot_data.get('win_rate')
        )

        self.execute_query(query, params)

    def record_risk_event(self, event_data: Dict):
        """Record risk management event"""

        query = '''
            INSERT INTO risk_events 
            (event_type, severity, symbol, description, value, threshold, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''

        params = (
            event_data['event_type'],
            event_data['severity'],
            event_data.get('symbol'),
            event_data['description'],
            event_data.get('value'),
            event_data.get('threshold'),
            event_data.get('timestamp', datetime.now())
        )

        self.execute_query(query, params)

    def get_portfolio_performance(self, days_back: int = 30) -> Dict:
        """Get portfolio performance metrics"""

        cutoff_date = datetime.now() - timedelta(days=days_back)

        query = '''
            SELECT timestamp, total_value, daily_pnl, sharpe_ratio, max_drawdown
            FROM portfolio_snapshots 
            WHERE timestamp >= ?
            ORDER BY timestamp
        '''

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[cutoff_date])

        if df.empty:
            return {}

        # Calculate performance metrics
        total_return = (df['total_value'].iloc[-1] - df['total_value'].iloc[0]) / df['total_value'].iloc[0]
        volatility = df['daily_pnl'].std() * np.sqrt(252) if len(df) > 1 else 0
        max_drawdown = df['max_drawdown'].min() if 'max_drawdown' in df.columns else 0

        return {
            'total_return': total_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'current_value': df['total_value'].iloc[-1],
            'data_points': len(df)
        }

    def get_strategy_performance(self, strategy_name: str = None, days_back: int = 30) -> pd.DataFrame:
        """Get strategy performance history"""

        cutoff_date = datetime.now() - timedelta(days=days_back)

        query = '''
            SELECT strategy_name, symbol, date, total_signals, profitable_signals,
                   total_pnl, sharpe_ratio, max_drawdown, win_rate
            FROM strategy_performance 
            WHERE date >= ?
        '''

        params = [cutoff_date]

        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)

        query += " ORDER BY date DESC"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        return df

    def _track_query_performance(self, query: str, execution_time: float):
        """Track query performance for optimization"""

        # Extract query type (first word)
        query_type = query.strip().split()[0].upper()

        with self.stats_lock:
            if query_type not in self.query_stats:
                self.query_stats[query_type] = {
                    'count': 0,
                    'total_time': 0,
                    'avg_time': 0,
                    'max_time': 0
                }

            stats = self.query_stats[query_type]
            stats['count'] += 1
            stats['total_time'] += execution_time
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['max_time'] = max(stats['max_time'], execution_time)

    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""

        stats = {}

        with self.get_connection() as conn:
            # Table sizes
            tables = [
                'market_data', 'strategy_signals', 'trades', 'portfolio_snapshots',
                'strategy_performance', 'risk_events', 'news_sentiment'
            ]

            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[f'{table}_count'] = count

            # Database file size
            stats['database_size_mb'] = os.path.getsize(self.config.db_path) / (1024 * 1024)

            # Latest data timestamps
            latest_market_data = conn.execute(
                "SELECT MAX(timestamp) FROM market_data"
            ).fetchone()[0]
            stats['latest_market_data'] = latest_market_data

            # Query performance stats
            stats['query_performance'] = dict(self.query_stats)

        return stats

    def create_backup(self, backup_name: str = None) -> str:
        """Create database backup"""

        if not backup_name:
            backup_name = f"trading_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_file = os.path.join(self.config.backup_path, f"{backup_name}.db")

        # Create backup directory
        os.makedirs(self.config.backup_path, exist_ok=True)

        # Copy database file
        shutil.copy2(self.config.db_path, backup_file)

        # Compress if enabled
        if self.config.compression:
            zip_file = f"{backup_file}.zip"
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(backup_file, os.path.basename(backup_file))

            # Remove uncompressed backup
            os.remove(backup_file)
            backup_file = zip_file

        self.logger.info(f"Database backup created: {backup_file}")
        return backup_file

    def restore_backup(self, backup_file: str):
        """Restore database from backup"""

        # Stop background tasks during restore
        self.stop_background_tasks()

        try:
            # Close all connections
            with self.pool_lock:
                for conn in self.connection_pool:
                    conn.close()
                self.connection_pool.clear()

            # Handle compressed backups
            if backup_file.endswith('.zip'):
                with zipfile.ZipFile(backup_file, 'r') as zf:
                    zf.extractall(self.config.backup_path)
                    # Find the extracted database file
                    extracted_files = zf.namelist()
                    db_file = next((f for f in extracted_files if f.endswith('.db')), None)
                    if db_file:
                        backup_db_path = os.path.join(self.config.backup_path, db_file)
                    else:
                        raise ValueError("No .db file found in backup zip")
            else:
                backup_db_path = backup_file

            # Replace current database
            shutil.copy2(backup_db_path, self.config.db_path)

            # Clean up temporary files
            if backup_file.endswith('.zip') and os.path.exists(backup_db_path):
                os.remove(backup_db_path)

            self.logger.info(f"Database restored from: {backup_file}")

        finally:
            # Restart background tasks
            self._start_background_tasks()

    def cleanup_old_data(self, days_to_keep: int = None):
        """Clean up old data based on retention policy"""

        if days_to_keep is None:
            days_to_keep = self.config.cleanup_days

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        # Tables to clean up with their timestamp columns
        cleanup_tables = [
            ('market_data', 'timestamp'),
            ('strategy_signals', 'timestamp'),
            ('news_sentiment', 'publish_time'),
            ('risk_events', 'timestamp')
        ]

        total_deleted = 0

        for table, timestamp_col in cleanup_tables:
            query = f"DELETE FROM {table} WHERE {timestamp_col} < ?"
            deleted = self.execute_query(query, (cutoff_date,))
            total_deleted += deleted
            self.logger.info(f"Cleaned up {deleted} old records from {table}")

        # Vacuum database to reclaim space
        with self.get_connection() as conn:
            conn.execute("VACUUM")

        self.logger.info(f"Cleanup complete. Total records deleted: {total_deleted}")
        return total_deleted

    def optimize_database(self):
        """Optimize database performance"""

        with self.get_connection() as conn:
            # Update table statistics
            conn.execute("ANALYZE")

            # Rebuild indexes
            conn.execute("REINDEX")

            # Optimize WAL file
            if self.config.wal_mode:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        self.logger.info("Database optimization complete")

    def check_integrity(self) -> Dict[str, Any]:
        """Check database integrity"""

        results = {}

        with self.get_connection() as conn:
            # SQLite integrity check
            integrity_result = conn.execute("PRAGMA integrity_check").fetchone()[0]
            results['sqlite_integrity'] = integrity_result

            # Check for orphaned records
            orphaned_signals = conn.execute('''
                SELECT COUNT(*) FROM strategy_signals s
                WHERE NOT EXISTS (
                    SELECT 1 FROM market_data m 
                    WHERE m.symbol = s.symbol 
                    AND date(m.timestamp) = date(s.timestamp)
                )
            ''').fetchone()[0]
            results['orphaned_signals'] = orphaned_signals

            # Check for missing indexes
            expected_indexes = [
                'idx_market_data_symbol_time',
                'idx_signals_strategy_time',
                'idx_trades_symbol_time'
            ]

            existing_indexes = [
                row[1] for row in conn.execute(
                    "SELECT * FROM sqlite_master WHERE type='index'"
                ).fetchall()
            ]

            missing_indexes = [idx for idx in expected_indexes if idx not in existing_indexes]
            results['missing_indexes'] = missing_indexes

            # Data quality checks
            duplicate_trades = conn.execute('''
                SELECT COUNT(*) FROM (
                    SELECT trade_id, COUNT(*) as cnt 
                    FROM trades 
                    GROUP BY trade_id 
                    HAVING cnt > 1
                )
            ''').fetchone()[0]
            results['duplicate_trades'] = duplicate_trades

            # Check for data consistency
            negative_prices = conn.execute('''
                SELECT COUNT(*) FROM market_data 
                WHERE close_price <= 0 OR open_price <= 0
            ''').fetchone()[0]
            results['negative_prices'] = negative_prices

        results['overall_status'] = 'HEALTHY' if all([
            integrity_result == 'ok',
            orphaned_signals == 0,
            len(missing_indexes) == 0,
            duplicate_trades == 0,
            negative_prices == 0
        ]) else 'ISSUES_FOUND'

        return results

    def export_data(self, table_name: str, output_format: str = 'csv',
                    start_date: datetime = None, end_date: datetime = None) -> str:
        """Export data to various formats"""

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{table_name}_export_{timestamp}.{output_format}"
        output_path = os.path.join('exports', output_file)

        # Ensure export directory exists
        os.makedirs('exports', exist_ok=True)

        # Build query based on table
        if table_name == 'market_data':
            query = '''
                SELECT symbol, timestamp, open_price, high_price, low_price, 
                       close_price, volume, source
                FROM market_data
            '''
            date_column = 'timestamp'
        elif table_name == 'trades':
            query = '''
                SELECT trade_id, symbol, strategy_name, action, quantity, price,
                       commission, timestamp, pnl, status
                FROM trades
            '''
            date_column = 'timestamp'
        elif table_name == 'portfolio_snapshots':
            query = '''
                SELECT timestamp, total_value, cash_balance, invested_value,
                       daily_pnl, total_pnl, sharpe_ratio, max_drawdown
                FROM portfolio_snapshots
            '''
            date_column = 'timestamp'
        else:
            raise ValueError(f"Export not supported for table: {table_name}")

        # Add date filters if provided
        params = []
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append(f"{date_column} >= ?")
                params.append(start_date)
            if end_date:
                conditions.append(f"{date_column} <= ?")
                params.append(end_date)

            query += " WHERE " + " AND ".join(conditions)

        query += f" ORDER BY {date_column}"

        # Execute query and export
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if output_format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif output_format.lower() == 'json':
            df.to_json(output_path, orient='records', date_format='iso')
        elif output_format.lower() == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")

        self.logger.info(f"Data exported to: {output_path}")
        return output_path

    def _start_background_tasks(self):
        """Start background maintenance tasks"""

        if self.background_running:
            return

        self.background_running = True

        def background_worker():
            while self.background_running:
                try:
                    # Create periodic backup
                    if datetime.now().hour == 2:  # 2 AM backup
                        self.create_backup()

                    # Cleanup old backups (keep last 7 days)
                    self._cleanup_old_backups()

                    # Optimize database (weekly)
                    if datetime.now().weekday() == 0 and datetime.now().hour == 3:  # Monday 3 AM
                        self.optimize_database()

                    # Sleep for 1 hour
                    time.sleep(3600)

                except Exception as e:
                    self.logger.error(f"Background task error: {e}")
                    time.sleep(300)  # Sleep 5 minutes on error

        self.backup_thread = threading.Thread(target=background_worker, daemon=True)
        self.backup_thread.start()

        self.logger.info("Background database tasks started")

    def stop_background_tasks(self):
        """Stop background maintenance tasks"""
        self.background_running = False
        if self.backup_thread and self.backup_thread.is_alive():
            self.backup_thread.join(timeout=5)
        self.logger.info("Background database tasks stopped")

    def _cleanup_old_backups(self):
        """Clean up old backup files"""

        if not os.path.exists(self.config.backup_path):
            return

        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days

        for filename in os.listdir(self.config.backup_path):
            filepath = os.path.join(self.config.backup_path, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                try:
                    os.remove(filepath)
                    self.logger.info(f"Removed old backup: {filename}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove backup {filename}: {e}")

    def get_data_summary(self, symbol: str = None, days_back: int = 7) -> Dict:
        """Get summary of available data"""

        cutoff_date = datetime.now() - timedelta(days=days_back)

        summary = {}

        with self.get_connection() as conn:
            # Market data summary
            if symbol:
                market_data_count = conn.execute(
                    "SELECT COUNT(*) FROM market_data WHERE symbol = ? AND timestamp >= ?",
                    (symbol, cutoff_date)
                ).fetchone()[0]

                latest_price = conn.execute(
                    "SELECT close_price FROM market_data WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
                    (symbol,)
                ).fetchone()

                summary['market_data_points'] = market_data_count
                summary['latest_price'] = latest_price[0] if latest_price else None

                # Strategy signals for symbol
                signals_count = conn.execute(
                    "SELECT COUNT(*) FROM strategy_signals WHERE symbol = ? AND timestamp >= ?",
                    (symbol, cutoff_date)
                ).fetchone()[0]
                summary['strategy_signals'] = signals_count

                # Trades for symbol
                trades_count = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE symbol = ? AND timestamp >= ?",
                    (symbol, cutoff_date)
                ).fetchone()[0]
                summary['trades_count'] = trades_count

            else:
                # Overall summary
                total_symbols = conn.execute(
                    "SELECT COUNT(DISTINCT symbol) FROM market_data WHERE timestamp >= ?",
                    (cutoff_date,)
                ).fetchone()[0]
                summary['total_symbols'] = total_symbols

                total_data_points = conn.execute(
                    "SELECT COUNT(*) FROM market_data WHERE timestamp >= ?",
                    (cutoff_date,)
                ).fetchone()[0]
                summary['total_data_points'] = total_data_points

                total_signals = conn.execute(
                    "SELECT COUNT(*) FROM strategy_signals WHERE timestamp >= ?",
                    (cutoff_date,)
                ).fetchone()[0]
                summary['total_signals'] = total_signals

                total_trades = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE timestamp >= ?",
                    (cutoff_date,)
                ).fetchone()[0]
                summary['total_trades'] = total_trades

        return summary

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_background_tasks()

        # Close all pooled connections
        with self.pool_lock:
            for conn in self.connection_pool:
                try:
                    conn.close()
                except:
                    pass
            self.connection_pool.clear()


# Integration example with your working_gui.py
class TradingDatabaseInterface:
    """
    High-level interface for integrating database with trading system
    Provides simplified methods for common trading operations
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)

    def log_strategy_signal(self, strategy_name: str, symbol: str,
                            action: str, confidence: float, price: float):
        """Log a strategy signal"""

        signal_data = {
            'strategy_name': strategy_name,
            'symbol': symbol,
            'signal_type': action,
            'confidence': confidence,
            'entry_price': price,
            'timestamp': datetime.now()
        }

        self.db.record_strategy_signal(signal_data)

    def log_trade_execution(self, symbol: str, action: str, quantity: int,
                            price: float, strategy_name: str = None) -> str:
        """Log a trade execution and return trade ID"""

        trade_data = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'strategy_name': strategy_name,
            'timestamp': datetime.now()
        }

        return self.db.record_trade(trade_data)

    def update_portfolio_snapshot(self, positions: Dict[str, int],
                                  cash: float, current_prices: Dict[str, float]):
        """Update portfolio snapshot with current state"""

        # Calculate portfolio value
        invested_value = sum(qty * current_prices.get(symbol, 0)
                             for symbol, qty in positions.items())
        total_value = cash + invested_value

        # Get previous snapshot for P&L calculation
        previous = self.db.execute_query(
            "SELECT total_value FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1",
            fetch='one'
        )

        daily_pnl = total_value - previous[0] if previous else 0

        snapshot_data = {
            'total_value': total_value,
            'cash_balance': cash,
            'invested_value': invested_value,
            'positions': positions,
            'daily_pnl': daily_pnl,
            'timestamp': datetime.now()
        }

        self.db.record_portfolio_snapshot(snapshot_data)

    def get_recent_performance(self, days: int = 30) -> Dict:
        """Get recent performance metrics"""
        return self.db.get_portfolio_performance(days)

    def log_risk_event(self, event_type: str, description: str,
                       severity: str = "MEDIUM", symbol: str = None):
        """Log a risk management event"""

        event_data = {
            'event_type': event_type,
            'severity': severity,
            'symbol': symbol,
            'description': description,
            'timestamp': datetime.now()
        }

        self.db.record_risk_event(event_data)


# Example usage
if __name__ == "__main__":
    # Initialize database
    config = DatabaseConfig(
        db_path="nyx_trading.db",
        backup_interval_hours=6,
        cleanup_days=90
    )

    db_manager = DatabaseManager(config)

    # Create high-level interface
    trading_db = TradingDatabaseInterface(db_manager)

    # Example operations
    print("NYX Trading Database System")
    print("=" * 40)

    # Log some sample data
    trading_db.log_strategy_signal(
        strategy_name="MA_Crossover",
        symbol="AAPL",
        action="BUY",
        confidence=0.85,
        price=150.25
    )

    trade_id = trading_db.log_trade_execution(
        symbol="AAPL",
        action="BUY",
        quantity=100,
        price=150.30,
        strategy_name="MA_Crossover"
    )

    print(f"Trade logged with ID: {trade_id}")

    # Update portfolio
    trading_db.update_portfolio_snapshot(
        positions={"AAPL": 100, "GOOGL": 50},
        cash=85000.0,
        current_prices={"AAPL": 151.00, "GOOGL": 140.50}
    )

    # Get database stats
    stats = db_manager.get_database_stats()
    print(f"\nDatabase Statistics:")
    for key, value in stats.items():
        if not key.endswith('_count'):
            continue
        print(f"  {key}: {value}")

    # Check integrity
    integrity = db_manager.check_integrity()
    print(f"\nDatabase Integrity: {integrity['overall_status']}")

    # Create backup
    backup_file = db_manager.create_backup()
    print(f"Backup created: {backup_file}")

    print(f"\n✅ Database system ready for trading operations!")