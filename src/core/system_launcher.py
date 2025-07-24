import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

class TradingSystemLauncher:
    def __init__(self):
        print('ImRogue Trading System Launcher')

    def run_system_checks(self):
        print('Running system checks...')

        # Check API config
        if Path('config/.env.api').exists():
            print('OK API configuration found')
        else:
            print('FAIL API configuration missing')
            return False

        # Test API client
        try:
            from components.data_sources.multi_api_client import MultiAPIClient
            client = MultiAPIClient()
            quote = client.get_quote('SPY')
            if quote:
                print('OK API connectivity test passed')
                return True
            else:
                print('FAIL API connectivity test failed')
                return False
        except Exception as e:
            print('FAIL API test error:', str(e))
            return False

    def launch_system(self):
        print('Starting ImRogue Trading System...')

        if not self.run_system_checks():
            print('System checks failed!')
            return False

        print('System ready!')
        print('Available commands: status, test, quote, help, quit')

        while True:
            try:
                cmd = input('ImRogue> ').strip().lower()

                if cmd == 'quit':
                    break
                elif cmd == 'status':
                    print('System status: Running')
                elif cmd == 'test':
                    print('Running quick test...')
                    try:
                        from components.data_sources.multi_api_client import MultiAPIClient
                        client = MultiAPIClient()
                        quote = client.get_quote('SPY')
                        if quote:
                            print('Test passed - SPY price available')
                        else:
                            print('Test failed')
                    except Exception as e:
                        print('Test error:', str(e))
                elif cmd == 'quote':
                    symbol = input('Enter symbol: ').strip().upper()
                    if symbol:
                        try:
                            from components.data_sources.multi_api_client import MultiAPIClient
                            client = MultiAPIClient()
                            quote = client.get_quote(symbol)
                            if quote:
                                print(symbol + ': $' + str(quote.get('price', 'N/A')))
                            else:
                                print('No quote available')
                        except Exception as e:
                            print('Quote error:', str(e))
                elif cmd == 'help':
                    print('Commands: status, test, quote, help, quit')
                else:
                    print('Unknown command. Type help for available commands.')

            except (EOFError, KeyboardInterrupt):
                break

        print('Shutting down system...')
        return True

def main():
    launcher = TradingSystemLauncher()
    launcher.launch_system()

if __name__ == '__main__':
    main()
