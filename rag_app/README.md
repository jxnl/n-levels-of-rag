# Introduction

This is a README detailing how to use `rag-app`, a simple cli which can be used to interact with a local lancedb database.


# Commands

## Querying the Database

We can use the command `rag-app db query-db <db path> <db table> <query>` in order to query our database. We use the `text-embedding-3` model with a dimensionality of 256 by default.

```
(venv) admin@admins-mac-mini n-levels-of-rag % rag-app db query-db "./db" paul_graham "What is the most important thing to consider when working on hard problems?"
=========================
Chunk 1
=========================
start to get something done.  One of the biggest mistakes ambitious people make is to allow setbacks to destroy their
morale all at once, like a balloon bursting. You can inoculate yourself against this by explicitly considering setbacks
a part of your process. Solving hard problems always involves some backtracking.  Doing great work is a depth-first
search whose root node is the desire to. So "If at first you don't succeed, try, try again" isn't quite right. It should
be: If at first you don't succeed, either try again, or backtrack and then try again.  "Never give up" is also not quite
right. Obviously there are times when it's the right choice to eject. A more precise version would be: Never let
setbacks panic you into backtracking more than you need to. Corollary: Never abandon the root node.  It's not
necessarily a bad sign if work is a struggle, any more than it's a bad sign to be out of breath while running. It
depends how fast you're running. So learn to distinguish good pain from bad. Good pain is a
=========================
Chunk 2
=========================
if there's an important but overlooked problem in your neighborhood, it's probably already on your subconscious radar
screen. So try asking yourself: if you were going to take a break from "serious" work to work on something just because
it would be really interesting, what would you do? The answer is probably more important than it seems.  Originality in
choosing problems seems to matter even more than originality in solving them. That's what distinguishes the people who
discover whole new fields. So what might seem to be merely the initial step — deciding what to work on — is in a sense
the key to the whole game.  Few grasp this. One of the biggest misconceptions about new ideas is about the ratio of
question to answer in their composition. People think big ideas are answers, but often the real insight was in the
question.  Part of the reason we underrate questions is the way they're used in schools. In schools they tend to exist
only briefly before being answered, like unstable particles. But a really good
=========================
```
