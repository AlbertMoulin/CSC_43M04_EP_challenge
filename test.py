"""
Analyze the finite channel set to create powerful channel-based features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def analyze_finite_channels(train_path="dataset/train_val.csv", test_path="dataset/test.csv"):
    """Analyze the finite channel set for maximum predictive power."""
    
    print("🔍 Analyzing Finite Channel Set...")
    
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Training videos: {len(train_df):,}")
    print(f"Test videos: {len(test_df):,}")
    
    # Channel analysis
    train_channels = set(train_df['channel'].unique())
    test_channels = set(test_df['channel'].unique())
    
    print(f"Training channels: {len(train_channels):,}")
    print(f"Test channels: {len(test_channels):,}")
    print(f"Overlap: {len(train_channels & test_channels):,}")
    print(f"Test-only channels: {len(test_channels - train_channels):,}")
    print(f"Train-only channels: {len(train_channels - train_channels):,}")
    
    # This is CRITICAL - if all test channels are in training, we can use historical performance!
    overlap_ratio = len(train_channels & test_channels) / len(test_channels)
    print(f"📊 Channel overlap ratio: {overlap_ratio:.1%}")
    
    if overlap_ratio > 0.95:
        print("🚀 HUGE ADVANTAGE: Almost all test channels have training history!")
    
    # Compute detailed channel statistics
    channel_stats = train_df.groupby('channel').agg({
        'views': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'title': [
            lambda x: x.str.len().mean(),  # avg title length
            lambda x: x.str.lower().str.contains('vfx|cgi|breakdown').mean(),  # vfx ratio
            lambda x: x.str.lower().str.contains('short film|animated short').mean(),  # short film ratio
            lambda x: x.str.contains('"').mean(),  # quote ratio
            lambda x: x.str.contains(':').mean(),  # colon ratio
        ],
        'description': [
            lambda x: x.fillna('').str.len().mean(),  # avg desc length
            lambda x: x.fillna('').str.lower().str.contains('subscribe|like|share').mean(),  # CTA ratio
            lambda x: x.fillna('').str.lower().str.contains('facebook|twitter|instagram').mean(),  # social ratio
        ],
        'date': [
            lambda x: pd.to_datetime(x).dt.hour.mean(),  # avg posting hour
            lambda x: pd.to_datetime(x).dt.dayofweek.mean(),  # avg day of week
        ]
    }).round(3)
    
    # Flatten column names
    channel_stats.columns = [
        'video_count', 'avg_views', 'median_views', 'std_views', 'min_views', 'max_views',
        'avg_title_length', 'vfx_ratio', 'short_film_ratio', 'quote_ratio', 'colon_ratio',
        'avg_desc_length', 'cta_ratio', 'social_ratio',
        'avg_posting_hour', 'avg_posting_dow'
    ]
    
    # Add derived features
    channel_stats['consistency'] = 1 - (channel_stats['std_views'] / channel_stats['avg_views']).fillna(0)
    channel_stats['view_range'] = channel_stats['max_views'] - channel_stats['min_views']
    channel_stats['reliability'] = np.minimum(channel_stats['video_count'] / 50, 1.0)  # Confidence based on video count
    
    # Percentile rankings (this is key for finite sets!)
    for col in ['avg_views', 'median_views', 'video_count', 'vfx_ratio', 'short_film_ratio', 'cta_ratio', 'social_ratio']:
        channel_stats[f'{col}_percentile'] = channel_stats[col].rank(pct=True)
    
    print(f"\n📈 Channel Performance Tiers:")
    
    # Create performance tiers
    channel_stats['performance_tier'] = pd.cut(
        channel_stats['avg_views'], 
        bins=[0, 10000, 100000, 1000000, float('inf')],
        labels=['Low', 'Medium', 'High', 'Elite']
    )
    
    tier_counts = channel_stats['performance_tier'].value_counts()
    for tier, count in tier_counts.items():
        avg_views = channel_stats[channel_stats['performance_tier'] == tier]['avg_views'].mean()
        print(f"  {tier}: {count} channels, avg {avg_views:,.0f} views")
    
    # Top predictive features for each tier
    print(f"\n🎯 Top Channel Features by Tier:")
    for tier in ['Elite', 'High', 'Medium', 'Low']:
        tier_data = channel_stats[channel_stats['performance_tier'] == tier]
        if len(tier_data) > 0:
            print(f"\n{tier} tier ({len(tier_data)} channels):")
            print(f"  VFX ratio: {tier_data['vfx_ratio'].mean():.2f}")
            print(f"  Short film ratio: {tier_data['short_film_ratio'].mean():.2f}")
            print(f"  CTA ratio: {tier_data['cta_ratio'].mean():.2f}")
            print(f"  Social ratio: {tier_data['social_ratio'].mean():.2f}")
            print(f"  Consistency: {tier_data['consistency'].mean():.2f}")
    
    # Save channel lookup table
    channel_stats.to_csv('channel_lookup_table.csv')
    print(f"\n💾 Saved channel lookup table with {len(channel_stats)} channels")
    
    # Analyze test set coverage
    print(f"\n🔍 Test Set Channel Analysis:")
    test_channel_stats = []
    unknown_channels = []
    
    for channel in test_channels:
        if channel in channel_stats.index:
            stats = channel_stats.loc[channel]
            test_channel_stats.append({
                'channel': channel,
                'train_video_count': stats['video_count'],
                'avg_views': stats['avg_views'],
                'performance_tier': stats['performance_tier']
            })
        else:
            unknown_channels.append(channel)
    
    test_coverage_df = pd.DataFrame(test_channel_stats)
    
    if len(test_coverage_df) > 0:
        print(f"Known channels in test: {len(test_coverage_df)}")
        print(f"Test videos from known channels: ~{len(test_coverage_df) / len(test_channels) * len(test_df):.0f}")
        print(f"Tier distribution in test:")
        print(test_coverage_df['performance_tier'].value_counts())
    
    if unknown_channels:
        print(f"⚠️ Unknown channels in test: {len(unknown_channels)}")
        print(f"Unknown channels: {unknown_channels[:5]}...")
    
    return channel_stats, test_coverage_df


def create_channel_prediction_baseline(channel_stats):
    """Create a simple channel-based baseline predictor."""
    print(f"\n🎯 Channel-Based Prediction Baseline:")
    
    # Simple baseline: predict channel average
    baseline_predictions = {}
    
    for channel, stats in channel_stats.iterrows():
        # Use median for more robust predictions
        baseline_predictions[channel] = stats['median_views']
    
    # For unknown channels, use overall median
    overall_median = channel_stats['median_views'].median()
    baseline_predictions['<UNKNOWN>'] = overall_median
    
    print(f"Channel baseline predictor created for {len(baseline_predictions)} channels")
    print(f"Prediction range: {min(baseline_predictions.values()):,.0f} - {max(baseline_predictions.values()):,.0f}")
    print(f"Unknown channel default: {overall_median:,.0f}")
    
    return baseline_predictions


if __name__ == "__main__":
    channel_stats, test_coverage = analyze_finite_channels()
    baseline = create_channel_prediction_baseline(channel_stats)
    
    print(f"\n✅ Analysis complete! Key takeaways:")
    print(f"1. Channel lookup table saved with rich features")
    print(f"2. Most test channels have training history")
    print(f"3. Channel-based baseline predictor ready")
    print(f"4. Performance tiers identified for adaptive modeling")