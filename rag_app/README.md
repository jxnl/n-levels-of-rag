# Introduction

This is a README detailing how to use `rag-app`, a simple cli which can be used to interact with a local lancedb database. You can install this by running the command

```
pip3 install -e .
```


# Commands

## Querying the Database

We can use the command `rag-app query db <db path> <db table> <query>` in order to query our database. We use the `text-embedding-3` model with a dimensionality of 256 by default.

```
>> rag-app query db ./db paul_graham "What's the biggest challenge facing any startup"                                                             
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃            ┃                                                                                                                            ┃
┃  Chunk ID  ┃  Result                                                                                                                    ┃
┃            ┃                                                                                                                            ┃
┣━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃            ┃                                                                                                                            ┃
┃  57        ┃  o ideas fit together like two puzzle pieces.  How do you get from starting small to doing something great? By making      ┃
┃            ┃  successive versions. Great things are almost always made in successive versions. You start with something small and       ┃
┃            ┃  evolve it, and the final version is both cleverer and more ambitious than anything you could have planned.  It's          ┃
┃            ┃  particularly useful to make successive versions when you're making something for people — to get an initial version in    ┃
┃            ┃  front of them quickly, and then evolve it based on their response.  Begin by trying the simplest thing that could         ┃
┃            ┃  possibly work. Surprisingly often, it does. If it doesn't, this will at least get you started.  Don't try to cram too     ┃
┃            ┃  much new stuff into any one version. There are names for doing this with the first version (taking too long to ship) and  ┃
┃            ┃  the second (the second system effect), but these are both merely instances of a more general principle.  An early         ┃
┃            ┃  version of a new project will sometimes be dismissed as a toy. It's a good sign when people                               ┃
┃            ┃                                                                                                                            ┃
┣━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃            ┃                                                                                                                            ┃
┃  17        ┃  fields where you have to be independent-minded to succeed — where your ideas have to be not just correct, but novel as    ┃
┃            ┃  well.  This is obviously the case in science. You can't publish papers saying things that other people have already       ┃
┃            ┃  said. But it's just as true in investing, for example. It's only useful to believe that a company will do well if most    ┃
┃            ┃  other investors don't; if everyone else thinks the company will do well, then its stock price will already reflect that,  ┃
┃            ┃  and there's no room to make money.  What else can we learn from these fields? In all of them you have to put in the       ┃
┃            ┃  initial effort. Superlinear returns seem small at first. At this rate, you find yourself thinking, I'll never get         ┃
┃            ┃  anywhere. But because the reward curve rises so steeply at the far end, it's worth taking extraordinary measures to get   ┃
┃            ┃  there.  In the startup world, the name for this principle is "do things that don't scale." If you pay a ridiculous        ┃
┃            ┃  amount of attention to your tiny initial set of customers, ideally you'll kick off e                                      ┃
┃            ┃                                                                                                                            ┃
┣━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
```
