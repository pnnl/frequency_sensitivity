import re

def floating_string(x:float or str):
    """convert a float to a decent string for file-path usage, or vice versa"""
    if isinstance(x, float):
        x: str = f'{x:.3e}'
        x = x.replace('.', 'pt')
        return x
    else:
        x = '.'.join(x.split('pt'))
        if re.search('e', x):
            return float(x.split('e')[0])*(10**float(x.split('e')[1]))
        else:
            return float(x)

def chunkify(l: list, num_chunks:int, chunk:int):
    """break a list into chunks (useful for splitting a loop across multiple
    processes, e.g. across multiple slurm jobs)"""
    start = chunk*(len(l)//num_chunks)
    stop = ((chunk+1)*(len(l)//num_chunks) if chunk < num_chunks -1 else None)
    return l[start:stop]