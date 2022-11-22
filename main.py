import re

if __name__ == '__main__':
    X_train = [['Glazed', 'Finger', 'Wings'], ['Country', 'Scalloped', 'Potatoes', '&amp;', 'Ham', '(Crock', 'Pot)'], ['Fruit', 'Dream', 'Cookies'], ['Tropical', 'Breakfast', 'Risotti'], ['Linguine', 'W/', 'Olive,', 'Anchovy', 'and', 'Tuna', 'Sauce']]
    Y_train = [['chicken-wings', 'sugar,', 'cornstarch', 'salt', 'ground ginger', 'pepper', 'water', 'lemon juice', 'soy sauce'], ['potatoes', 'onion', 'cooked ham', 'country gravy mix', 'cream of mushroom soup', 'water', 'cheddar cheese'], ['butter', 'shortening', 'granulated sugar', 'eggs', 'baking soda', 'baking powder', 'vanilla', 'all-purpose flour', 'white chocolate chips', 'orange drink mix', 'colored crystal sugar'], ['water', 'instant brown rice', 'pineapple tidbits', 'skim evaporated milk', 'raisins', 'sweetened flaked coconut', 'toasted sliced almonds', 'banana'], ['anchovy fillets', 'tuna packed in oil', 'kalamata olive', 'garlic cloves', 'fresh parsley', 'fresh lemon juice', 'salt %26 pepper', 'olive oil', 'linguine']]

    for x, y in zip(X_train, Y_train):
        print(x, y)
