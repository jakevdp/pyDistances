import os, sys

def _parse_src(infile, outfile):
    def_dict = {}
    out_lines = []

    lines = open(infile).readlines()

    OF = open(outfile, 'w')

    OF.write('### auto-generated from %s using %s\n\n' % (infile, __file__))
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('@DEFINE'):
            key = line.split()[1]
            val = []
            i += 1
            while not lines[i].startswith('@ENDDEF'):
                val.append(lines[i])
                i += 1
            def_dict[key] = val
            i += 1
            continue
        
        elif '@' in line:
            linesplit = line.split('@')
            found = False
            for key in def_dict:
                if linesplit[1].startswith(key):
                    OF.write(linesplit[0].join([''] + def_dict[key]))
                    found = True
                    break
            if found:
                i += 1
                continue
        
        OF.write(line)
        i += 1

if __name__ == '__main__':
    infile = 'distfuncs.pxi_src'
    outfile = 'distfuncs.pxi'
    _parse_src(infile, outfile)
            
    
