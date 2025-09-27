import os, sys, time
from reddit_scrape import EnhancedRedditScraper
from cluster import train_and_save

def cfg():
    return {"host":os.getenv("MYSQL_HOST","localhost"),"user":os.getenv("MYSQL_USER","root"),"password":os.getenv("MYSQL_PASSWORD","1234"),"database":os.getenv("MYSQL_DB","reddit_data")}

def main():
    if len(sys.argv)<2:
        print("usage: python automation.py <interval_minutes> [posts] [clusters]")
        return
    interval=float(sys.argv[1]); posts=int(sys.argv[2]) if len(sys.argv)>2 else 200; clusters=int(sys.argv[3]) if len(sys.argv)>3 else 3
    sub=os.getenv("SUBREDDIT_URL","https://www.reddit.com/r/tech/new")
    scraper=EnhancedRedditScraper(sub,cfg())
    while True:
        try:
            print("[*] fetching"); data=scraper.scrape(num_posts=posts); scraper.display_summary(data)
            print("[*] training"); train_and_save(cfg(), n_clusters=clusters, outdir="artifacts")
            print("[*] done")
        except Exception as e:
            print("[!] error:",e)
        time.sleep(interval*60)

if __name__=="__main__":
    main()

