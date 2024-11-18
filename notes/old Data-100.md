[The latest note for Data100 sp22](./Data100%20sp22.md)
# Data sampling and Probability
Data Science Lifecycle:
![](https://lh7-us.googleusercontent.com/oV2Ot23dqxOg68EbzcD_y8adLNfVtUbm_mSuw4dUQNmG7qts5AD8f7zvYJbKY-QiY6wAtbj56kV-P3VKdpDGcppPIVKgmLi_xF_bzPv2vAUMokvH_Kx7Sq46_Pm8p8N0HUkjv0kom1bVibZaqg=s2048)
- Censeus(普查)
- Survey:set of questions

## Samples:
Samples are often used to make inferences about the popultation
How you draw the sample will affect your accuracy

- Two common sources of error:
		**chance error** : random samples can vary from what is expected,in any direction
		**bias** : a systematic error in one direction
		
*Convenience samples* : A convenience sample is whoever you can get a hold of
	- Not a good idea for inference!
	- Haphazard ≠ random.
	- Sources of bias can introduce themselves in ways you may not think of!

*Quota samples* : you first specify your desired breakdown of various subgroups, and then reach those targets however you can.
	- For example: you may want to sample individuals in your town, and you may want the age distribution of your sample to match that of your town’s census results.

Convenience samples and Quota samples are not random

**sample is represeative of the population**	
	- Don't just try to get a big sample
	- If your method of sampling is bad, and your sample is big. you will have a **big, bad sample**
	
## Bias
- Population, samples, and sampling frame
	- **Population**: The group that you want to learn something about.
	- **Sampling Frame**: The list from which the sample is drawn.
		- If you’re sampling people, the sampling frame is the set of all people that could possibly end up in your sample.
	- **Sample**: Who you actually end up sampling. 
	    - A subset of your sampling frame.

- Common Biases:

	**Selection Bias**
	- Systematically excluding (or favoring) particular groups. 
	- How to avoid: Examine the sampling frame and the method of sampling.
	
	**Response Bias**
	- People don’t always respond truthfully。
	- How to avoid: Examine the nature of questions and the method of surveying.

	**Non-response Bias**
	- People don’t always respond.    
	- How to avoid: Keep your surveys short, and be persistent. 
	- People who don’t respond aren’t like the people who do!
## Probability sampling
Why
-  Random samples can produce biased estimates of population characteristic
-  But with random samples we are able to **estimate the bias and chance error**

Probability samples and random samples will sometimes mean the same thing.

In order for a sample to be a probability sample:
- You must be able to provide the chance that any specified set of individuals will be in the sample.
- All individuals in the population do not need to have the same chance of being selected.
- You will still be able to measure the errors, because you know all the probabilities.

A **random sample with replacement** is a sample drawn uniformly at random with repalcement

A **simple random sample is a sample** drawn uniformly at random without replacement

Approximation
- A common situation in data science
	- we have enormous population
	- we can only afford to sample a relatively small number of individuals
- If the **population is huge** compared to the sample, **then random sampling with and without replacement are pretty much the same**.
- **Probabilities of sampling with replacement are much easier to compute!**

## Binomial and multinomial probabilities

Binomial and multinomial probabilities arise when we :
- Sample at random, with replacement
- Sample a fixed number times
- Sample from a categorical distribution
- Want to count the number of each category that end up in our sample. 

# Random Variables
## Random Variables
A random variable is a **numerical function of a random sample**

Functions of random variables:
- A function of random variable is also a random variable
- If you create multiple random variables based on your sample, then functions of them are also random variables

## Distribution
Random variables have a finite number of possible values
$$
P(X = x)
$$
Probability distributions largely fall into two main categories
- Discrete 
- Continuous 

## Equality
Consider two random variables X and Y based on our sample.

- X and Y are **equal** if, for every sample s, X(s) = Y(x)

- X and Y are **identically distributed** if the distribution of X is the same as the distribution of Y

## Summary

- Random variables are functions of our sample.

- The expectation of a random variable is the weighted average of its possible values, weighted by the probabilities of those values.
	- Expectation behaves nicely with linear transformations of random variables.  
	- Expectation is also additive.

# SQL
## Databases

- A database is an organization collection of data
- A database management system (DBMS) is software system that stores, manages, and facilitates access to one or more databases.
### Advantages

Data Storage:
- Reliable storage to survive system crashes and disk failures.
- Optimize to compute on data that does not fit in memory.
- Special data structures to improve performance (see CS (W)186).

Data Management:

- Configure how data is logically organized and who has access.
- Can enforce guarantees on the data (e.g. non-negative bank account balance).
- Can be used to prevent data anomalies.

Ensures safe concurrent operations on data.

## SQL

![](https://lh7-us.googleusercontent.com/slidesz/AGV_vUeZg0no-8XFzHEh_Zhxrs-Z42WsFkhpBWlN3oP26zOBhre7GXzz2gqqLtmKe4BAp5bDpsYeVomiZ3DnaFGy8eJJrm3VKWtlwGFBqX1xRO1u9qtHnqsJBMzoYajLSzP-vjtyAMWwcWTGS3Mz2kxI2cJN=s2048?key=BxCECP5D-TFSe1CGMgCy0A)

### Joins

- Cross Join
	All pairs of rows appear in the result
 - Inner Join
	Only pairs of matching rows appear in the result
- Left Outer Join
	Every row in the first table appears inthe
- Right Outer Join 
	Every row in the second table appears in the result, matching or not
- Full Outer Join 
	Every row in both tables appears, matching or not.
- Other Join Conditions 
	We can join on conditions other than equality.
	
![](https://lh7-us.googleusercontent.com/slidesz/AGV_vUdN9PLPtOLd6AGt64gTxRO9EbhiLcKZGdqPPe2xTTL_cAlr-ggTITV3ECEOBR_Rh-6dnxXGHnpugiF6MwHZ3XvANKQE0O1COLlRl8R1x619vzyBMhBtB5OwXOkH9uuP_oXsUroje870PMwcSuQCgMc0=s2048?key=BxCECP5D-TFSe1CGMgCy0A)

### Null Values

- Field values are sometimes unknown
	- SQL provides a special value NULL for such situations    
	- Every data type can be NULL  

- The presence of null complicates many issues. E.g.:
	- Selection predicates (WHERE)
	- Aggregation

- But NULLs are common after outer joins

### SQL Predicates and Casting 

- SQL Predicates 
	In addition to numerical comparisons (=, <, >), SQL has built-in predicates.
- SQL Casting 
	Can use CAST to convert fields from one type to another:

### SQL Sampling, Subqueries, and  Common Table Expressions


# Pandas

![](https://lh7-us.googleusercontent.com/slidesz/AGV_vUdHcs6TbBwp-ULJTjqzfV6Y1cRplYn1p7LsGD3tOvyJ5tCe86e1wsyTpqR1vzm0ykXKnLZcWe-AiNgPXW6hmtL12YATg1HSRRc2boHF2qKZgCjbc9xXisV0A2dsdA_KVnzJnugRN8HLVWBa8vr6xyw=s2048?key=bMC6YRjlgFSC2bVLxQvH0Q)