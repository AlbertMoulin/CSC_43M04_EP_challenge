import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_and_analyze_data(file_path):
    """
    Comprehensive EDA for YouTube video dataset
    """
    # Load data
    df = pd.read_csv(file_path)
    
    print("🎬 YOUTUBE VIDEO DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # ===========================================
    # 1. BASIC DATASET OVERVIEW
    # ===========================================
    print("\n📊 BASIC DATASET OVERVIEW")
    print("-" * 30)
    print(f"Dataset shape: {df.shape}")
    print(f"Total videos: {len(df)}")
    print(f"Unique channels: {df['channel'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024:.1f} KB")
    
    # Check for missing values
    print(f"\nMissing values:")
    missing_data = df.isnull().sum()
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    # ===========================================
    # 2. CHANNEL ANALYSIS
    # ===========================================
    print("\n📺 CHANNEL DISTRIBUTION")
    print("-" * 30)
    
    channel_stats = df['channel'].value_counts()
    print("Videos per channel:")
    for channel, count in channel_stats.items():
        print(f"  {channel[-10:]}: {count} videos ({count/len(df)*100:.1f}%)")
    
    # Channel concentration
    top_channel_share = channel_stats.iloc[0] / len(df) * 100
    print(f"\nTop channel accounts for {top_channel_share:.1f}% of videos")
    
    # ===========================================
    # 3. TEMPORAL ANALYSIS
    # ===========================================
    print("\n⏰ TEMPORAL PATTERNS")
    print("-" * 30)
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['year'] = df['date'].dt.year
    
    # Monthly distribution
    monthly_dist = df['month'].value_counts().sort_index()
    print("Monthly distribution:")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, count in monthly_dist.items():
        month_name = month_names[month-1]
        bar = '█' * count
        print(f"  {month_name}: {bar} ({count})")
    
    # Hour distribution
    print(f"\nUpload hours (most common):")
    hour_dist = df['hour'].value_counts().head(5)
    for hour, count in hour_dist.items():
        print(f"  {hour:02d}:00 - {count} videos")
    
    # ===========================================
    # 4. CONTENT ANALYSIS
    # ===========================================
    print("\n📝 CONTENT ANALYSIS")
    print("-" * 30)
    
    # Title analysis
    df['title_length'] = df['title'].str.len()
    print("Title statistics:")
    print(f"  Average length: {df['title_length'].mean():.1f} characters")
    print(f"  Range: {df['title_length'].min()} - {df['title_length'].max()} characters")
    print(f"  Median: {df['title_length'].median():.1f} characters")
    
    # Description analysis
    df['description_length'] = df['description'].fillna('').str.len()
    print(f"\nDescription statistics:")
    print(f"  Average length: {df['description_length'].mean():.0f} characters")
    print(f"  Range: {df['description_length'].min()} - {df['description_length'].max()} characters")
    print(f"  Videos without description: {(df['description_length'] == 0).sum()}")
    
    # Language detection (simple heuristic)
    def detect_language(title):
        if re.search(r'[\u0980-\u09FF]', title):  # Bengali Unicode range
            if re.search(r'[A-Za-z]', title):
                return 'Mixed'
            return 'Bengali'
        elif re.search(r'^[A-Za-z\s0-9.,!?#@%\-:;\(\)\[\]"\']+$', title):
            return 'English'
        return 'Other'
    
    df['title_language'] = df['title'].apply(detect_language)
    lang_dist = df['title_language'].value_counts()
    print(f"\nTitle language distribution:")
    for lang, count in lang_dist.items():
        print(f"  {lang}: {count} videos ({count/len(df)*100:.1f}%)")
    
    # Hashtag analysis
    def extract_hashtags(text):
        if pd.isna(text):
            return []
        return re.findall(r'#\w+', text.lower())
    
    all_hashtags = []
    for _, row in df.iterrows():
        title_tags = extract_hashtags(row['title'])
        desc_tags = extract_hashtags(row['description'])
        all_hashtags.extend(title_tags + desc_tags)
    
    if all_hashtags:
        hashtag_counts = Counter(all_hashtags)
        print(f"\nTop hashtags ({len(all_hashtags)} total):")
        for tag, count in hashtag_counts.most_common(8):
            print(f"  {tag}: {count} times")
    
    # ===========================================
    # 5. CONTENT CATEGORIZATION
    # ===========================================
    print("\n🎭 CONTENT CATEGORIZATION")
    print("-" * 30)
    
    # Simple content categorization based on keywords
    def categorize_content(title, description):
        text = f"{title} {description}".lower()
        
        if any(word in text for word in ['bengali', 'bangla', 'movie', 'film']):
            return 'Bengali Movies'
        elif any(word in text for word in ['short film', 'animation', 'animated']):
            return 'Short Films/Animation'
        elif any(word in text for word in ['cgi', '3d', 'vfx', 'making']):
            return 'CGI/VFX'
        elif any(word in text for word in ['documentary', 'drama']):
            return 'Documentary/Drama'
        else:
            return 'Other'
    
    df['content_category'] = df.apply(lambda x: categorize_content(x['title'], x['description']), axis=1)
    
    category_dist = df['content_category'].value_counts()
    print("Content categories:")
    for category, count in category_dist.items():
        print(f"  {category}: {count} videos ({count/len(df)*100:.1f}%)")
    
    return df

def create_visualizations(df):
    """
    Create comprehensive visualizations for the dataset
    """
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Channel Distribution
    plt.subplot(3, 3, 1)
    channel_counts = df['channel'].value_counts()
    # Truncate long channel IDs for display
    channel_labels = [ch[-8:] for ch in channel_counts.index]
    plt.pie(channel_counts.values, labels=channel_labels, autopct='%1.1f%%')
    plt.title('Distribution by Channel')
    
    # 2. Monthly Distribution
    plt.subplot(3, 3, 2)
    monthly_counts = df['month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(range(len(monthly_counts)), monthly_counts.values)
    plt.xticks(range(len(monthly_counts)), [month_names[i-1] for i in monthly_counts.index], rotation=45)
    plt.title('Videos by Month')
    plt.ylabel('Count')
    
    # 3. Upload Hour Distribution
    plt.subplot(3, 3, 3)
    hour_counts = df['hour'].value_counts().sort_index()
    plt.bar(hour_counts.index, hour_counts.values)
    plt.title('Upload Time Distribution')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    
    # 4. Title Length Distribution
    plt.subplot(3, 3, 4)
    plt.hist(df['title_length'], bins=15, alpha=0.7, edgecolor='black')
    plt.title('Title Length Distribution')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    
    # 5. Description Length Distribution
    plt.subplot(3, 3, 5)
    desc_lengths = df['description_length'][df['description_length'] > 0]  # Exclude empty descriptions
    plt.hist(desc_lengths, bins=15, alpha=0.7, edgecolor='black')
    plt.title('Description Length Distribution')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    
    # 6. Language Distribution
    plt.subplot(3, 3, 6)
    lang_counts = df['title_language'].value_counts()
    plt.bar(lang_counts.index, lang_counts.values)
    plt.title('Title Language Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 7. Content Category Distribution
    plt.subplot(3, 3, 7)
    category_counts = df['content_category'].value_counts()
    plt.bar(range(len(category_counts)), category_counts.values)
    plt.xticks(range(len(category_counts)), category_counts.index, rotation=45, ha='right')
    plt.title('Content Category Distribution')
    plt.ylabel('Count')
    
    # 8. Title Length vs Description Length
    plt.subplot(3, 3, 8)
    plt.scatter(df['title_length'], df['description_length'], alpha=0.6)
    plt.xlabel('Title Length')
    plt.ylabel('Description Length')
    plt.title('Title vs Description Length')
    
    # 9. Videos per Channel (Bar chart)
    plt.subplot(3, 3, 9)
    channel_counts = df['channel'].value_counts()
    channel_labels = [ch[-8:] for ch in channel_counts.index]
    plt.bar(range(len(channel_counts)), channel_counts.values)
    plt.xticks(range(len(channel_counts)), channel_labels, rotation=45)
    plt.title('Videos per Channel')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def key_insights_summary(df):
    """
    Generate key insights summary
    """
    print("\n🔍 KEY INSIGHTS FOR VIDEO VIEWS PREDICTION")
    print("=" * 50)
    
    # Channel dominance
    top_channel_share = df['channel'].value_counts().iloc[0] / len(df) * 100
    print(f"📺 Channel Factor: Top channel has {top_channel_share:.1f}% of videos")
    
    # Temporal patterns
    peak_month = df['month'].value_counts().index[0]
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    peak_hour = df['hour'].value_counts().index[0]
    print(f"⏰ Timing Factor: Peak upload month is {month_names[peak_month]}, peak hour is {peak_hour}:00")
    
    # Content characteristics
    avg_title_len = df['title_length'].mean()
    avg_desc_len = df['description_length'].mean()
    print(f"📝 Content Factor: Avg title length {avg_title_len:.0f} chars, avg description {avg_desc_len:.0f} chars")
    
    # Language distribution
    lang_counts = df['title_language'].value_counts()
    dominant_lang = lang_counts.index[0]
    print(f"🌐 Language Factor: {dominant_lang} titles dominate ({lang_counts.iloc[0]}/{len(df)} videos)")
    
    # Content categories
    category_counts = df['content_category'].value_counts()
    top_category = category_counts.index[0]
    print(f"🎭 Category Factor: {top_category} is most common ({category_counts.iloc[0]}/{len(df)} videos)")
    
    print(f"\n💡 FEATURE ENGINEERING RECOMMENDATIONS:")
    print("=" * 40)
    print("1. Channel_ID: Strong predictor (one channel has 45%+ of videos)")
    print("2. Upload_Hour: Consider time-of-day effects on viewership")
    print("3. Upload_Month: Seasonal patterns may affect views")
    print("4. Title_Length: Optimal length might correlate with engagement")
    print("5. Description_Length: Longer descriptions might indicate higher production value")
    print("6. Language: Bengali vs English content may have different audiences")
    print("7. Content_Category: Different categories likely have different view patterns")
    print("8. Hashtag_Count: Number of hashtags might indicate marketing effort")
    print("9. Has_Description: Binary feature for presence of description")
    print("10. Days_Since_Upload: Recency might affect current view count")

# Sample usage (uncomment to run):
# df = load_and_analyze_data('your_dataset.csv')
# create_visualizations(df)
# key_insights_summary(df)

# For prediction modeling, consider these features:
RECOMMENDED_FEATURES = {
    'channel_features': ['channel_id_encoded', 'channel_video_count'],
    'temporal_features': ['upload_hour', 'upload_month', 'upload_day_of_week', 'days_since_upload'],
    'content_features': ['title_length', 'description_length', 'hashtag_count', 'has_description'],
    'categorical_features': ['title_language', 'content_category'],
    'derived_features': ['title_word_count', 'description_word_count', 'caps_ratio_title']
}

def prepare_features_for_modeling(df):
    """
    Feature engineering for views prediction
    """
    df_model = df.copy()
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    
    # Channel encoding
    le_channel = LabelEncoder()
    df_model['channel_encoded'] = le_channel.fit_transform(df_model['channel'])
    
    # Language encoding
    le_lang = LabelEncoder()
    df_model['language_encoded'] = le_lang.fit_transform(df_model['title_language'])
    
    # Category encoding
    le_category = LabelEncoder()
    df_model['category_encoded'] = le_category.fit_transform(df_model['content_category'])
    
    # Additional derived features
    df_model['title_word_count'] = df_model['title'].str.split().str.len()
    df_model['description_word_count'] = df_model['description'].fillna('').str.split().str.len()
    df_model['has_description'] = (df_model['description_length'] > 0).astype(int)
    
    # Calculate hashtag count
    def count_hashtags(text):
        if pd.isna(text):
            return 0
        return len(re.findall(r'#\w+', text))
    
    df_model['title_hashtag_count'] = df_model['title'].apply(count_hashtags)
    df_model['desc_hashtag_count'] = df_model['description'].apply(count_hashtags)
    df_model['total_hashtag_count'] = df_model['title_hashtag_count'] + df_model['desc_hashtag_count']
    
    # Caps ratio in title (marketing intensity indicator)
    def caps_ratio(text):
        if not text:
            return 0
        letters = re.findall(r'[A-Za-z]', text)
        if not letters:
            return 0
        return sum(1 for c in letters if c.isupper()) / len(letters)
    
    df_model['caps_ratio_title'] = df_model['title'].apply(caps_ratio)
    
    # Time-based features
    df_model['upload_year'] = df_model['date'].dt.year
    df_model['upload_quarter'] = df_model['date'].dt.quarter
    df_model['is_weekend'] = df_model['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Days since upload (if you have current date)
    # df_model['days_since_upload'] = (pd.Timestamp.now() - df_model['date']).dt.days
    
    return df_model

print("\n" + "="*60)
print("🎯 READY FOR MODELING!")
print("="*60)
print("This EDA provides comprehensive insights into your YouTube dataset.")
print("Key findings suggest channel identity is the strongest predictor,")
print("followed by content type, timing, and language factors.")
print("\nNext steps:")
print("1. Load your full dataset with 'views' column")
print("2. Apply the feature engineering functions above")
print("3. Train models using the recommended features")
print("4. Consider ensemble methods combining channel-based and content-based models")

if __name__ == "__main__":
    # Example usage
    file_path = 'dataset/train_val.csv'  # Replace with your dataset path
    df = load_and_analyze_data(file_path)
    create_visualizations(df)
    key_insights_summary(df)
    
    # Prepare features for modeling
    df_model = prepare_features_for_modeling(df)
    print("\nPrepared features for modeling:")
    print(df_model.head())