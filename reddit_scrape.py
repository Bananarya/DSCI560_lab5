#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import mysql.connector
from mysql.connector import Error
import time
import re
from datetime import datetime
import logging
import hashlib
from collections import Counter
import string
from cluster import *

# Optional: For OCR functionality (uncomment if you want to use)
# import pytesseract
# from PIL import Image
# from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Common stop words for keyword extraction
STOP_WORDS = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                  'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                  'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                  'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'])

class EnhancedRedditScraper:
    def __init__(self, subreddit_url, db_config, enable_ocr=False):
        """
        Initialize the Enhanced Reddit scraper
        
        Args:
            subreddit_url: URL of the subreddit to scrape
            db_config: Database configuration dictionary
            enable_ocr: Boolean to enable OCR for image text extraction
        """
        self.base_url = subreddit_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.db_config = db_config
        self.enable_ocr = enable_ocr
        self.setup_database()
    
    def setup_database(self):
        """Create enhanced database and table structure"""
        conn = None
        try:
            # Connect without specifying database
            db_config_temp = self.db_config.copy()
            db_config_temp.pop('database', None)
            
            conn = mysql.connector.connect(**db_config_temp)
            cursor = conn.cursor()
            
            # Create database
            cursor.execute("CREATE DATABASE IF NOT EXISTS reddit_data")
            cursor.execute("USE reddit_data")
            
            # Create enhanced table with additional preprocessing fields
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reddit_posts_enhanced (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    post_id VARCHAR(255) UNIQUE,
                    title TEXT,
                    title_cleaned TEXT,
                    author VARCHAR(255),
                    author_masked VARCHAR(255),
                    score INT,
                    num_comments INT,
                    url TEXT,
                    content TEXT,
                    content_cleaned TEXT,
                    keywords TEXT,
                    topics VARCHAR(255),
                    image_text TEXT,
                    timestamp VARCHAR(255),
                    timestamp_converted DATETIME,
                    is_promoted BOOLEAN DEFAULT FALSE,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logging.info("Enhanced database setup completed")
            
        except Error as e:
            logging.error(f"Database setup error: {e}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
    
    def remove_html_tags(self, text):
        if not text:
            return ""
        
        # Parse with BeautifulSoup to remove HTML
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Additional cleanup for common HTML entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&quot;', '"')
        
        return text
    
    def clean_special_characters(self, text):
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', ' ', text)
        
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def mask_username(self, username):
        if not username or username == 'unknown':
            return 'anonymous'
        
        # Create a consistent hash for the same username
        hash_object = hashlib.md5(username.encode())
        hash_hex = hash_object.hexdigest()
        
        # Return masked format: user_[first 8 chars of hash]
        return f"user_{hash_hex[:8]}"
    
    def convert_timestamp(self, timestamp_str):
        if not timestamp_str:
            return None
        
        try:
            # Try parsing ISO format
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt
        except:
            try:
                # Try parsing Unix timestamp
                dt = datetime.fromtimestamp(int(timestamp_str))
                return dt
            except:
                return None
    
    def extract_keywords(self, text, num_keywords=5):
        if not text:
            return []
        
        # Convert to lowercase and remove punctuation
        text_lower = text.lower()
        translator = str.maketrans('', '', string.punctuation)
        text_clean = text_lower.translate(translator)
        
        # Split into words
        words = text_clean.split()
        
        # Filter out stop words and short words
        meaningful_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        
        # Count word frequency
        word_freq = Counter(meaningful_words)
        
        # Get top keywords
        top_keywords = [word for word, freq in word_freq.most_common(num_keywords)]
        
        return top_keywords
    
    def identify_topics(self, text, keywords):
        topics = []
        
        # Define topic patterns for r/tech
        topic_patterns = {
            'AI': ['artificial intelligence', 'machine learning', 'ai', 'neural', 'gpt', 'chatgpt', 'llm'],
            'Security': ['security', 'hack', 'breach', 'vulnerability', 'cyber', 'privacy', 'encryption'],
            'Mobile': ['phone', 'smartphone', 'android', 'ios', 'iphone', 'mobile', 'app'],
            'Gaming': ['game', 'gaming', 'console', 'playstation', 'xbox', 'nintendo', 'steam'],
            'Social Media': ['facebook', 'twitter', 'instagram', 'social media', 'tiktok', 'youtube'],
            'Hardware': ['cpu', 'gpu', 'processor', 'graphics', 'chip', 'semiconductor', 'nvidia', 'amd'],
            'Software': ['software', 'windows', 'linux', 'macos', 'operating system', 'update', 'bug'],
            'Internet': ['internet', 'broadband', 'wifi', 'network', '5g', 'isp', 'bandwidth'],
            'Crypto': ['crypto', 'bitcoin', 'blockchain', 'ethereum', 'nft', 'defi'],
            'Cloud': ['cloud', 'aws', 'azure', 'google cloud', 'saas', 'serverless']
        }
        
        text_lower = text.lower() if text else ""
        keywords_str = ' '.join(keywords).lower() if keywords else ""
        combined_text = text_lower + ' ' + keywords_str
        
        for topic, patterns in topic_patterns.items():
            for pattern in patterns:
                if pattern in combined_text:
                    topics.append(topic)
                    break
        
        return list(set(topics))  # Remove duplicates
    
    def check_if_promoted(self, post_data):
        promoted_indicators = ['promoted', 'sponsored', 'advertisement', 'ad', 'partner content']
        
        title_lower = (post_data.get('title', '') or '').lower()
        content_lower = (post_data.get('content', '') or '').lower()

        for indicator in promoted_indicators:
            if indicator in title_lower or indicator in content_lower:
                return True
        
        return False
    
    def extract_image_text(self, image_url):
        if not self.enable_ocr or not image_url:
            return ""
        
        try:
            # Uncomment if using pytesseract
            # response = self.session.get(image_url, timeout=10)
            # img = Image.open(BytesIO(response.content))
            # text = pytesseract.image_to_string(img)
            # return self.clean_special_characters(text)
            return ""
        except Exception as e:
            logging.warning(f"OCR extraction failed: {e}")
            return ""
    
    def preprocess_post(self, post_data):
        if not post_data:
            return None
        
        # Remove HTML tags
        post_data['title_cleaned'] = self.remove_html_tags(post_data.get('title', ''))
        post_data['content_cleaned'] = self.remove_html_tags(post_data.get('content', ''))
        
        # Clean special characters
        post_data['title_cleaned'] = self.clean_special_characters(post_data['title_cleaned'])
        post_data['content_cleaned'] = self.clean_special_characters(post_data['content_cleaned'])
        
        # Mask username for privacy
        post_data['author_masked'] = self.mask_username(post_data.get('author', ''))
        
        # Convert timestamp
        post_data['timestamp_converted'] = self.convert_timestamp(post_data.get('timestamp', ''))
        
        # Extract keywords
        combined_text = f"{post_data['title_cleaned']} {post_data['content_cleaned']}"
        keywords = self.extract_keywords(combined_text, num_keywords=7)
        post_data['keywords'] = ', '.join(keywords)
        
        # Identify topics
        topics = self.identify_topics(combined_text, keywords)
        post_data['topics'] = ', '.join(topics)
        
        # Check if promoted
        post_data['is_promoted'] = self.check_if_promoted(post_data)
        
        # Extract text from images (if any)
        if 'image_url' in post_data:
            post_data['image_text'] = self.extract_image_text(post_data['image_url'])
        else:
            post_data['image_text'] = ""
        
        return post_data
    
    def fetch_posts(self, num_posts=100):
        posts = []
        after = None
        posts_fetched = 0
        
        while posts_fetched < num_posts:
            try:
                url = f"{self.base_url}?after={after}" if after else self.base_url
                logging.info(f"Fetching from: {url}")
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find posts
                post_elements = soup.find_all('shreddit-post')
                if not post_elements:
                    post_elements = soup.find_all('div', {'data-testid': 'post-container'})
                if not post_elements:
                    post_elements = soup.find_all('div', class_=re.compile('Post|thing'))
                
                logging.info(f"Found {len(post_elements)} posts on this page")
                
                for post in post_elements:
                    if posts_fetched >= num_posts:
                        break
                    
                    post_data = self.extract_post_data(post)
                    if post_data:
                        # Apply preprocessing
                        post_data = self.preprocess_post(post_data)
                        
                        # Filter out promoted content unless specifically needed
                        if not post_data['is_promoted'] or True:  # Keep promoted for analysis
                            posts.append(post_data)
                            posts_fetched += 1
                
                # Get next page
                next_button = soup.find('a', {'rel': 'next'})
                if next_button and 'after=' in next_button.get('href', ''):
                    after = next_button['href'].split('after=')[1].split('&')[0]
                else:
                    last_post = post_elements[-1] if post_elements else None
                    if last_post:
                        after = last_post.get('id') or last_post.get('data-fullname')
                
                if not after or posts_fetched >= num_posts:
                    break
                
                time.sleep(2)  # Rate limiting
                
            except requests.RequestException as e:
                logging.error(f"Request error: {e}")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                break
        
        logging.info(f"Total posts fetched and preprocessed: {len(posts)}")
        return posts
    
    def extract_post_data(self, post_element):
        try:
            post_data = {}
            
            if post_element.name == 'shreddit-post':
                post_data['post_id'] = post_element.get('id', '')
                post_data['title'] = post_element.get('post-title', '')
                post_data['author'] = post_element.get('author', '')
                post_data['score'] = int(post_element.get('score', 0))
                post_data['num_comments'] = int(post_element.get('comment-count', 0))
                post_data['url'] = post_element.get('content-href', '')
                post_data['timestamp'] = post_element.get('created-timestamp', '')
            else:
                title_elem = post_element.find('h3') or post_element.find('a', class_=re.compile('title'))
                post_data['title'] = title_elem.text.strip() if title_elem else ''
                
                author_elem = post_element.find('a', href=re.compile(r'/user/'))
                post_data['author'] = author_elem.text.strip() if author_elem else 'unknown'
                
                score_elem = post_element.find('div', class_=re.compile('score'))
                score_text = score_elem.text.strip() if score_elem else '0'
                try:
                    post_data['score'] = int(re.sub(r'[^\d-]', '', score_text))
                except:
                    post_data['score'] = 0
                
                comments_elem = post_element.find('a', href=re.compile(r'/comments/'))
                if comments_elem:
                    match = re.search(r'(\d+)\s*comment', comments_elem.text)
                    post_data['num_comments'] = int(match.group(1)) if match else 0
                else:
                    post_data['num_comments'] = 0
                
                link_elem = post_element.find('a', href=re.compile(r'^https?://'))
                post_data['url'] = link_elem['href'] if link_elem else ''
                
                post_data['post_id'] = post_element.get('data-fullname', '') or post_element.get('id', '')
                
                time_elem = post_element.find('time')
                post_data['timestamp'] = time_elem.get('datetime', '') if time_elem else ''
            
            post_data['content'] = ''  # Would need to follow link for full content
            
            return post_data
            
        except Exception as e:
            logging.warning(f"Error extracting post data: {e}")
            return None
    
    def store_posts(self, posts):
        conn = None
        try:
            db_config_temp = self.db_config.copy()
            db_config_temp.pop('database', None)
            
            conn = mysql.connector.connect(**db_config_temp)
            cursor = conn.cursor()
            cursor.execute("USE reddit_data")
            
            stored_count = 0
            
            for post in posts:
                try:
                    query = """
                        INSERT INTO reddit_posts_enhanced 
                        (post_id, title, title_cleaned, author, author_masked, score, 
                         num_comments, url, content, content_cleaned, keywords, topics, 
                         image_text, timestamp, timestamp_converted, is_promoted)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        score = VALUES(score),
                        num_comments = VALUES(num_comments),
                        keywords = VALUES(keywords),
                        topics = VALUES(topics)
                    """
                    
                    values = (
                        post.get('post_id', ''),
                        post.get('title', ''),
                        post.get('title_cleaned', ''),
                        post.get('author', ''),
                        post.get('author_masked', ''),
                        post.get('score', 0),
                        post.get('num_comments', 0),
                        post.get('url', ''),
                        post.get('content', ''),
                        post.get('content_cleaned', ''),
                        post.get('keywords', ''),
                        post.get('topics', ''),
                        post.get('image_text', ''),
                        post.get('timestamp', ''),
                        post.get('timestamp_converted'),
                        post.get('is_promoted', False)
                    )
                    
                    cursor.execute(query, values)
                    stored_count += 1
                    
                except Error as e:
                    logging.warning(f"Error storing post: {e}")
                    continue
            
            conn.commit()
            logging.info(f"Successfully stored {stored_count} preprocessed posts")
            
        except Error as e:
            logging.error(f"Database error: {e}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
    
    def scrape(self, num_posts=100):
        """Main method to scrape and preprocess Reddit posts"""
        logging.info(f"Starting to scrape and preprocess {num_posts} posts from r/tech")
        
        posts = self.fetch_posts(num_posts)
        
        if posts:
            # Filter out promoted content from final storage (optional)
            non_promoted = [p for p in posts if not p['is_promoted']]
            promoted = [p for p in posts if p['is_promoted']]
            
            logging.info(f"Found {len(promoted)} promoted posts and {len(non_promoted)} regular posts")
            
            # Store all posts (including promoted for analysis)
            self.store_posts(posts)
            
            logging.info(f"Scraping and preprocessing completed. {len(posts)} posts processed.")
            
            return posts
        else:
            logging.warning("No posts were fetched")
            return []
    
    def display_summary(self, posts):
        if not posts:
            return
        
        print(f"\n{'='*60}")
        print("SCRAPING AND PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total posts processed: {len(posts)}")
        
        # Topic distribution
        all_topics = []
        for post in posts:
            topics = post.get('topics', '').split(', ')
            all_topics.extend([t for t in topics if t])
        
        topic_counts = Counter(all_topics)
        print(f"\nTop Topics Found:")
        for topic, count in topic_counts.most_common(5):
            print(f"  - {topic}: {count} posts")
        
        # Promoted content
        promoted_count = sum(1 for p in posts if p.get('is_promoted', False))
        print(f"\nPromoted/Ad Content: {promoted_count} posts ({promoted_count/len(posts)*100:.1f}%)")
        
        # Keywords
        all_keywords = []
        for post in posts:
            keywords = post.get('keywords', '').split(', ')
            all_keywords.extend([k for k in keywords if k])
        
        keyword_counts = Counter(all_keywords)
        print(f"\nTop Keywords:")
        for keyword, count in keyword_counts.most_common(10):
            print(f"  - {keyword}: {count} occurrences")
        
        print(f"\nAverage score: {sum(p.get('score', 0) for p in posts) / len(posts):.2f}")
        print(f"Average comments: {sum(p.get('num_comments', 0) for p in posts) / len(posts):.2f}")
        print(f"{'='*60}\n")


def main():
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',  # Replace with your MySQL username
        'password': '1234',  # Replace with your MySQL password
        'database': 'reddit_data'
    }
    
    # Reddit URL to scrape
    reddit_url = 'https://www.reddit.com/r/tech/new'
    
    # Get number of posts from user input
    try:
        num_posts = int(input("Enter the number of posts to fetch (default 100): ") or "100")
    except ValueError:
        num_posts = 100
        print("Invalid input. Using default value of 100 posts.")
    
    # Ask about OCR
    enable_ocr = input("Enable OCR for image text extraction? (y/n, default n): ").lower() == 'y'
    
    # Create scraper instance
    scraper = EnhancedRedditScraper(reddit_url, db_config, enable_ocr)
    
    # Run the scraper
    posts = scraper.scrape(num_posts)
    # Display summary
    scraper.display_summary(posts)
    cluster_and_visualization(db_config)

if __name__ == "__main__":
    main()
