import sys
from cluster import load_artifacts, infer_with_artifacts

def main():
    model,kmeans,pca,df,vectors=load_artifacts("artifacts")
    print("ready. type text, or 'quit'")
    while True:
        try:
            s=input("query> ").strip()
        except (EOFError,KeyboardInterrupt):
            s="quit"
        if not s:
            continue
        if s.lower() in {"quit","exit"}:
            break
        try:
            cid,hits,out=infer_with_artifacts(model,kmeans,pca,df,vectors,s,top_k=10,outdir="artifacts")
            print("cluster:",cid)
            for i,row in enumerate(hits.itertuples(index=False),1):
                m=row.message[:120]+("..." if len(row.message)>120 else "")
                print(f"{i:>2}. {m}")
            print("chart:",out)
        except Exception as e:
            print("error:",e)

if __name__=="__main__":
    main()