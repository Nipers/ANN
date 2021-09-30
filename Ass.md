Ass

P1

(a)

it returns the max element's index in the input array.

For the function mysteryRecursive, when l= r, it returns l, which is the biggest number's between A[l] and A[r], when r - l = 1, it still returns the bigger number's index.

Then we assume the conclusion is right for r and l that satisify r - l <= n, we can see when r - l = n + 1, i and j will be the biggest numbers' indexs in the left and right part, so finally the biggest number's index between A[l] and A[r]  will be returned. 

(b) 

It belongs to Divide and Conquer Algorithmic Paradigm 

(c)

C(1) = 0, C(2) = 1

C(n) = C(n - n/2) + C(n/2) + 1

(d)

C(n) = n - 1

(e)

O(n)



P2

(a) calculate the ratio of fee to size for every transaction,  try to put the transaction among all the unselected with highest ratio into the set I until no transaction can be put into I

pseudo code  

```
Algorithm Greedy(fee, size):
	n = fee.length
	max-heap h
    result = 0
    for i in range(n):
        ratio[i] = fee[i] / size[i]
        h.push (ratio[i],i)
    while h.size() >= 0 && b >= 0:
        (r, i) = h.pop()
        if size[i] <= b:
            b -= size[i]
            result += fee[i]	
    return result
```

complexity analysis  

the complexity of ratio calculation is O(n), the complexity of the heap-building is O(nlogn), the complexity of the last loop is O(nlogn), so the finally complexity is O(nlogn)

run on toy example

r1 = 4, r2 = 3, r3 = 3, r4 = 2.75, r5 = 2.6

so the selected transaction sequence is 1 -> 2 -> 3 -> 4 -> 5, and 1, 2, 3 is legal to put into the set, 

4 and 5 can't be selected because there is no enough room.

result's change  process is 4 -> 13 ->19 -> 19 -> 19

b's change process is 7 -> 4 -> 2 -> 2 -> 2

Final result is 19

(b)

We use a two-dimension array to record answer.

result(i, j) means the biggest profit we can get using at most size i by selecting the former j transactions. so result(i, 0) = result (0, j) = 0

Then we can get the recursion formula
$$
result(i, j) = max(result(i, j - 1),result(i - size[j], j - 1) + fee[j])
$$
We need two-layer loop to fill the table, the first loop's  variable is j, and the second variable is i

j should increase from 1 to n, i should decrease from b to 1

table

| j/i  | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1    | 4    | 4    | 4    | 4    | 4    | 4    | 4    | 4    |
| 2    | 4    | 4    | 9    | 13   | 13   | 13   | 13   | 13   |
| 3    | 4    | 6    | 10   | 13   | 15   | 19   | 19   | 19   |
| 4    | 4    | 6    | 10   | 13   | 15   | 19   | 21   | 24   |
| 5    | 4    | 6    | 10   | 13   | 15   | 19   | 21   | 24   |

So the maximum profit is 24 and the set is 1, 2, 4

