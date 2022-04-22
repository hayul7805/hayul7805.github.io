# Introduction to Python Sequences

> Before we start there are some key phrases that will be useful to understand:

- **Mutable**: This means that the item can be changed once it has been created. The opposite of this is immutable which means that once the object has been created it cannot be changed. This affects how the data structure can or cannot be used later on in your code.
- **Ordered**: This means that the way in which the items are stored will not be changed (unless explicitly stated) and can be accessed by knowing which order the item was stored in. The opposite of this is an unordered object where the items stored cannot be accessed by the order that they were placed in.
- **Indexable**: This means that items can be accessed based on the order in which they were added into the data structure using the index. This is only applicable for ordered objects because the order stays the same whenever it is called or used.

## **List**

- **Mutable**: They can be changed once they have been created.
- **Indexable**: We can access data based on where they are placed in the list.
- **Ordered:** They maintain their order when they are created.
- **Can contain duplicate records**: Data with the same value can be stored in a list.

## **Tuples**

The second type of data structure in Python that we often come across is that of the Tuple. These are similar to lists but have the following characteristics:

- **Immutable**: Once they are created they cannot be changed.
- **Ordered:** The order in which items are added is maintained.
- **Indexable:** We can access items based on their index.
- **Can contain duplicates:** items with the same value can be in this data structure.

## **Sets** { }

Following this, another similar data structure to lists again is the set. The main characteristics of this is:

- **Mutable**: They can be changed once they have been created.
- **Unordered:** The order doesn’t remain the same when you call them later on.
- **Unindexable:** You cannot access a certain item with a given index.
- **No duplicates allowed**: A set cannot contain a duplicate value.

```python
#create a set using curly brackets
fruits = {"apple", "banana", "cherry"}

#create a set using the set constructor
vege = set(("courgette", "potato", "aubergine"))

#print the results
print(fruits)
print(vege)

#out:
{'cherry', 'apple', 'banana'}
{'potato', 'aubergine', 'courgette'}
```

## Dictionary { }

Finally, another data structure that you would often come across in programming in Python would be that of a dictionary. These have the characteristics of:

- **Mutable**: They can be changed after they have been created.
- **Ordered**: The order which items are added to the dictionary is maintained.
- **Indexable**: We can access items based on their index (in this case their key)
- **Cannot contain duplicate values**: At least in terms of their key

While this is similar to a set, the main difference is that the `key` and the `value` are separated by `:` which ensures that a dictionary is created.

```python
new_dict = {"Name":"Peter Jones",
           "Age":28,
           "Occupation":"Data Scientist"}
print(new_dict)

#out:
{'Name': 'Peter Jones', 'Age': 28, 'Occupation': 'Data Scientist'}
```

