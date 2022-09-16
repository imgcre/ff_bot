import csv
import imp
import json
from typing import Dict
import click

@click.command()
@click.argument('input', type=click.File('rt', encoding='utf-8'))
@click.argument('output', type=click.File('wt', encoding='utf-8'))
def main(input, output):
    j: Dict[str, any] = json.load(input)
    writer = csv.writer(output)
    writer.writerow(['key', 0, 1])
    writer.writerow(['#', 'Item', 'Gil'])
    writer.writerow(['int32', 'Item', 'uint32'])
    items = list(j.items())
    items.sort(key=lambda x: x[1]['id'])
    for i, (k, v) in enumerate(items):
        writer.writerow([i, f'Item#{v["id"]}', v['gil']])
        ...

if __name__ == '__main__':
    main()