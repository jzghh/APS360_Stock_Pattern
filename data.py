import yfinance as yf
import mplfinance as mpf
import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_data(ticker, start, end, interval):
    """Downloads and preprocesses data."""
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    
    # Handle MultiIndex columns (yfinance 新版本格式)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    df = df.dropna()
    return df

def calculate_label(df_current, df_next):
    """
    Calculate label based on price change between current and next period.
    根据当前和下一期的价格变化计算标签
    """
    if df_next is None or len(df_next) < 5:
        return None
    
    current_close = df_current['Close'].iloc[-1]
    next_close = df_next['Close'].iloc[-1]
    
    # Handle scalar conversion
    if hasattr(current_close, 'item'):
        current_close = current_close.item()
    if hasattr(next_close, 'item'):
        next_close = next_close.item()
    
    pct_change = (next_close - current_close) / current_close
    
    # Thresholds for classification (可调整)
    UP_THRESHOLD = 0.03      # 涨幅 >= 3%
    DOWN_THRESHOLD = -0.03   # 跌幅 <= -3%
    
    if pct_change >= UP_THRESHOLD:
        return "Up"
    elif pct_change <= DOWN_THRESHOLD:
        return "Down"
    else:
        return "Sideways"

def generate_labeled_dataset(config):
    """
    Generate labeled dataset for YOLOv8 classification.
    生成用于 YOLOv8 分类的带标签数据集
    """
    tickers = config['tickers']
    start_date = config['start_date']
    end_date = config['end_date']
    interval = config['interval']
    output_dir = config.get('dataset_dir', 'dataset')
    train_ratio = config.get('train_ratio', 0.8)
    
    # Create directory structure for YOLOv8 classification
    # YOLOv8 分类数据集结构
    classes = ["Up", "Down", "Sideways"]
    for split in ["train", "val"]:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
    
    # Custom style for cleaner images
    mc = mpf.make_marketcolors(up='green', down='red', edge='i', wick='i', volume='in')
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle='', rc={'font.size': 0})
    
    print(f"Processing {len(tickers)} tickers...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Train/Val split: {train_ratio:.0%}/{1-train_ratio:.0%}")
    print("-" * 50)
    
    stats = {"Up": 0, "Down": 0, "Sideways": 0}
    
    for ticker in tqdm(tickers, desc="Tickers"):
        try:
            print(f"\nDownloading {interval} data for {ticker}...")
            df = get_data(ticker, start_date, end_date, interval)
            
            if len(df) == 0:
                print(f"  {ticker}: No data available")
                continue
            
            # Group by month
            df['YearMonth'] = df.index.to_period('M')
            grouped = list(df.groupby('YearMonth'))
            
            for i in range(len(grouped) - 1):  # -1 因为需要下一个月来计算标签
                year_month, df_month = grouped[i]
                _, df_next_month = grouped[i + 1]
                
                if len(df_month) < 5:
                    continue
                
                # Calculate label
                label = calculate_label(df_month, df_next_month)
                if label is None:
                    continue
                
                # Decide train or val (随机划分)
                split = "train" if random.random() < train_ratio else "val"
                
                # Generate filename
                filename = f"{ticker}_{year_month}.png"
                save_path = os.path.join(output_dir, split, label, filename)
                
                # Remove helper column for plotting
                df_subset = df_month.drop(columns=['YearMonth'])
                
                # Plot
                fig, axes = mpf.plot(
                    df_subset,
                    type='candle',
                    style=style,
                    volume=False,
                    axisoff=True,
                    returnfig=True,
                    figsize=(8, 8),
                    tight_layout=True
                )
                
                fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
                plt.close(fig)
                stats[label] += 1
                
        except Exception as e:
            print(f"  {ticker}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("Dataset generation complete!")
    print(f"  Up: {stats['Up']} images")
    print(f"  Down: {stats['Down']} images")
    print(f"  Sideways: {stats['Sideways']} images")
    print(f"  Total: {sum(stats.values())} images")
    print(f"\nSaved to: {output_dir}/")

def main():
    config = load_config()
    
    # Add dataset-specific config if not present
    if 'dataset_dir' not in config:
        config['dataset_dir'] = 'dataset'
    if 'train_ratio' not in config:
        config['train_ratio'] = 0.8
    
    generate_labeled_dataset(config)

if __name__ == "__main__":
    main()
