import json
import sys
import sqlite3
from progress.bar import Bar

def open_db(name):
    return sqlite3.connect(name)

def create_table(db):
   ddl = """
   CREATE TABLE IF NOT EXISTS tweets (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        hashtags TEXT NOT NULL
   );
   """
   c = db.cursor()
   c.execute(ddl)

def upsert_tweet(db, cursor, tweet):
    hashtag_text = [hashtag['text'] for hashtag in tweet['entities']['hashtags']]
    hashtag_as_str = ','.join(hashtag_text)
    #print(hashtag_as_str)
    sql = """
    INSERT INTO tweets (id, text, hashtags) VALUES (?,?,?)
    ON CONFLICT (id) DO NOTHING;
    """
    return cursor.execute(sql, (tweet['id'], tweet['text'], hashtag_as_str))

def main(paths):
    db = open_db('tweets.sqlite3')
    create_table(db)
    db.commit()
    db.close()
    
    with Bar('Processing', max=len(paths)) as bar:
        for p in paths:
            with open(p, 'r') as f:
                db = open_db('tweets.sqlite3')
                cursor = db.cursor()
                for line in f.read().splitlines():
                    if len(line) == 0:
                        continue
                    try:
                        d = json.loads(str(line))
                    except:
                        import pdb; pdb.set_trace()
                    if 'entities' in d and len(d['entities']['hashtags']) > 0:
                        upsert_tweet(db, cursor, d)
                cursor.close()
                db.commit()
                db.close()
            bar.next()
        
main(sys.argv[1:])
