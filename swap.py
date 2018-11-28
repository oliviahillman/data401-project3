def mirror_swap(move):
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m']
    numbers = (['1','2','3','4','5','6','7','8','9','10','11','12','13'])[::-1]
    move_letter = letters.index(move[:1])
    move_number = numbers.index(move[1:])
    new_letter = abs(12-move_number)
    new_number = abs(12-move_letter)
    return (letters[new_letter] + numbers[new_number])
    