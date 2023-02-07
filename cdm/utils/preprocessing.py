from ase.calculators.vasp import VaspChargeDensity

class VaspChargeDensity(VaspChargeDensity):
    def read(self, filename, read_augs = True):
        """Re-implemenation of ASE functionality
        The adjustments support the new, faster _read_chg

        """
        import ase.io.vasp as aiv
        fd = open(filename)
        self.atoms = []
        self.chg = []
        self.chgdiff = []
        self.aug = ''
        self.augdiff = ''
        while True:
            try:
                atoms = aiv.read_vasp(fd)
            except (IOError, ValueError, IndexError):
                # Probably an empty line, or we tried to read the
                # augmentation occupancies in CHGCAR
                break
            fd.readline()
            ngr = fd.readline().split()
            shape = (int(ngr[0]), int(ngr[1]), int(ngr[2]))
            chg = self._read_chg(fd, shape, atoms.get_volume())
            self.chg.append(chg)
            self.atoms.append(atoms)
            # Check if the file has a spin-polarized charge density part, and
            # if so, read it in.
            fl = fd.tell()

            if not read_augs:
                break

            # First check if the file has an augmentation charge part (CHGCAR
            # file.)
            line1 = fd.readline()
            if line1 == '':
                break
            elif line1.find('augmentation') != -1:
                augs = [line1]
                while True:
                    line2 = fd.readline()
                    if line2.split() == ngr:
                        self.aug = ''.join(augs)
                        augs = []
                        chgdiff = np.empty(ng)
                        self._read_chg(fd, chgdiff, atoms.get_volume())
                        self.chgdiff.append(chgdiff)
                    elif line2 == '':
                        break
                    else:
                        augs.append(line2)
                if len(self.aug) == 0:
                    self.aug = ''.join(augs)
                    augs = []
                else:
                    self.augdiff = ''.join(augs)
                    augs = []
            elif line1.split() == ngr:
                chgdiff = np.empty(ng)
                self._read_chg(fd, chgdiff, atoms.get_volume())
                self.chgdiff.append(chgdiff)
            else:
                fd.seek(fl)
        fd.close()
        self.aug_string_to_dict()
        
    def _read_chg(self, fobj, shape, volume):
        """Replaces ASE's method
        This implementation is approximately 2x faster.
        This is important because reading data from disk can
        be a limiting factor for training speed.
        
        """
        chg = np.fromfile(fobj, count = np.prod(shape), sep=' ')

        chg = chg.reshape(shape, order = 'F')
        chg /= volume

        return chg
        
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
