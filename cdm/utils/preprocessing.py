from ase.calculators.vasp import VaspChargeDensity

class VaspChargeDensity(VaspChargeDensity):
    def read(self, filename):
        super().read(filename)
        self.aug_string_to_dict()
        
    def write(self, filename):
        self.aug_dict_to_string
        super().write(filename)
        
    def aug_dict_to_string(self):
        texts = re.split('.{10}E.{3}', self.aug)
        augs = []
        for i in range(len(self.aug_dict)):
            augs = [*augs, *(self.aug_dict[str(i+1)])]
            
        out = texts[0]

        for text, number in zip(texts[1:], augs):
            if number > 0:
                number = f'{10*number:.6E}'
                number = number.split('.')
                number = number[0] + number[1]
                number = ' 0.' + number
            elif number == 0:
                number = ' 0.0000000E+00'
            else:
                number = f'{10*number:.6E}'
                number = number.split('.')
                number = number[0][1] + number[1]
                number = '-0.' + number

            out += number
            out += text
            
        self.aug = out
        
    def aug_string_to_dict(self):
        self.aug_dict = {}
        split = [x.split() for x in self.aug.split('augmentation occupancies')[1:]]
        for row in split:
            label = row[0]
            augmentations = [float(x) for x in row[2:]]
            self.aug_dict[label] = augmentations
            
    def zero_augmentations(self):
        '''
        Set all augmentation occupancies to zero
        '''
        for key, value in self.aug_dict.items():
            value = [0 for i in value]
            self.aug_dict[key] = value
        
    def __eq__(self, other):
        if self.aug_dict != other.aug_dict:
            return False
        if self.atoms != other.atoms:
            return False
        
        for self_chg, other_chg in zip(self.chg, other.chg):
            if not (self_chg == other_chg).all():
                return False
            
        return True