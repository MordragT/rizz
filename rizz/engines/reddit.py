import praw
import gradio as gr
import pandas as pd
import os

SUBREDDIT = "beta"
STRATEGY = "hot"
SUBMISSION_COUNT = 5
COMMENTS_SORT_BY = "top"
COMMENTS_COUNT = 5

class Comment:
    def __init__(self, comment: praw.reddit.Comment):
        self.comment = comment

    def author(self) -> str:
        author = self.comment.author.name if self.comment.author is not None else "unknown"
        return author
    
    def body(self) -> str:
        return self.comment.body
    
    def score(self) -> str:
        return self.comment.score
    
    def markdown(self) -> str:
        return f"#### {self.author()} {self.score()}\n{self.body()}"

class Submission:
    def __init__(self, submission: praw.reddit.Submission):
        self.submission = submission

    def comments(self, count: int, sort_by: str) -> list[Comment]:
        self.submission.comment_sort = sort_by
        self.submission.comments.replace_more(limit=count)
        return [Comment(comment) for comment in self.submission.comments][:count]
    
    def id(self) -> str:
        return self.submission.id

    def title(self) -> str:
        return self.submission.title

    def author(self) -> str:
        author = self.submission.author.name if self.submission.author is not None else "unknown"
        return author
    
    def score(self) -> int:
        return self.submission.score
    
    def url(self) -> str:
        return self.submission.url
    
    def markdown(self) -> str:
        return f"## {self.author()} {self.score()}\n### [{self.title()}]({self.url()})"

class Subreddit:
    def __init__(self, subreddit: praw.reddit.Subreddit):
        self.subreddit = subreddit

    def new(self, count: int) -> list:
        ids = self.subreddit.new(limit=count)
        return list(ids)
    
    def top(self, count: int) -> list:
        ids = self.subreddit.top(limit=count)
        return list(ids)

    def hot(self, count: int) -> list:
        ids = self.subreddit.hot(limit=count)
        return list(ids)

    def rising(self, count: int) -> list:
        ids = self.subreddit.rising(limit=count)
        return list(ids)

    def controversial(self, count: int) -> list:
        ids = self.subreddit.controversial(limit=count)
        return list(ids)
    
    def name(self) -> str:
        return self.subreddit.display_name
    
    def description(self) -> str:
        return self.subreddit.description
    
    def markdown(self) -> str:
        return f"# {self.name()}\n{self.description()}"

class Reddit:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id = os.getenv("REDDIT_CLIENT_ID"),
            client_secret = os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent = "test",
            username = os.getenv("REDDIT_USERNAME"),
            password = os.getenv("REDDIT_PASSWORD"),
        )

    def subreddit(self, name: str) -> Subreddit:
        subreddit = self.reddit.subreddit(name)
        return Subreddit(subreddit)
    
    def submission(self, id) -> Submission:
        submission = self.reddit.submission(id)
        return Submission(submission)
    
    def submissions(self, ids: list) -> list[Submission]:
        return [self.submission(id) for id in ids]

class RedditEngine:
    def __init__(self, df):
        self.df = df
        self.reddit = Reddit()

    def render(self):
        with gr.Row():
            with gr.Column():
                subreddit = gr.Textbox(value=SUBREDDIT, label="Subreddit")
                strategy = gr.Dropdown(["new", "top", "hot", "controversial", "rising"], value=STRATEGY, allow_custom_value=False, label="Strategy")
                submission_count = gr.Number(SUBMISSION_COUNT, label="Submission Count")
                fetch = gr.Button("Fetch Submissions")

            with gr.Column():
                submission = gr.Dropdown(label="Submission", interactive=False, allow_custom_value=False)
                comments_sort_by = gr.Dropdown(["new", "top", "hot", "controversial", "rising"], value=COMMENTS_SORT_BY, allow_custom_value=False, label="Comments Sortby")
                comments_count = gr.Number(COMMENTS_COUNT, label="Comments Count")
                submit = gr.Button("Submit", interactive=False)

        submission_display = gr.Markdown()

        def on_fetch(name, strategy, n):
            subreddit = self.reddit.subreddit(name)

            ids = []
            match strategy:
                case "new":
                    ids = subreddit.new(n)
                case "top":
                    ids = subreddit.top(n)
                case "hot":
                    ids = subreddit.hot(n)
                case "controversial":
                    ids = subreddit.controversial(n)
                case "rising":
                    ids = subreddit.rising(n)

            submissions = self.reddit.submissions(ids)
            choices = [(s.title(), s.id()) for s in submissions]
            value = choices[0][1]

            return (
                gr.update(choices=choices, value=value, interactive=True),
                gr.update(value=submissions[0].markdown()),
                gr.update(interactive=True)
            )
        fetch.click(on_fetch, [subreddit, strategy, submission_count], [submission, submission_display, submit])


        def on_submission_change(id):
            submission = self.reddit.submission(id)
            return gr.update(value=submission.markdown())
        submission.change(on_submission_change, submission, submission_display)

        # def on_comment_count_change(i, n):
        #     submission = self.submissions[i]
        #     comments = [c.markdown() for c in submission.comments(n)]
        #     markdown = "\n".join(comments)
        #     return gr.update(value=markdown)
        # comment_count.change(on_comment_count_change, [submission, comment_count], comments_display)

        def on_submit(id, n, sort_by):
            submission = self.reddit.submission(id)
            comments = [[c.author(), c.score(), c.body()] for c in submission.comments(n, sort_by)]
            table = [[submission.author(), submission.score(), submission.title()]] + comments
            return pd.DataFrame.from_records(table, columns=["author", "score", "body"])
        submit.click(on_submit, [submission, comments_count, comments_sort_by], self.df)

        # def on_change(i, submissions):
        #     submission = submissions[i]
        #     comments = [c["body"] for c in submission["comments"]]
        #     units = [submission["title"]] + comments
        #     return gr.update(value=units)
        # submission.change(on_change, [submission, submissions], output)