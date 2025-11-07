#!/usr/bin/env python3
import argparse
from sympy import symbols, sympify, simplify_logic
from sympy.logic.boolalg import And, Or, Not, BooleanTrue, BooleanFalse
 
def parse_bnet_file(filename):
    """
    Parses a .bnet file where each non-comment line is of the form:
      Node, BooleanExpression
    Returns a list of (node, expression) tuples.
    """
    nodes = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            node = parts[0].strip()
            expr = ",".join(parts[1:]).strip()
            nodes.append((node, expr))
    return nodes

def get_regulators(expr, global_symbols):
    """
    Given a sympy Boolean expression and a global_symbols mapping,
    return the list of regulator symbols present in expr ordered by global index.
    """
    regs = [s for s in expr.free_symbols if s.name in global_symbols]
    regs.sort(key=lambda s: global_symbols[s.name]['index'])
    return regs

def term_to_cover(term, regulators):
    """
    Convert a single product term into a cover string of length equal to the number of regulators.
    For each regulator:
      - '1' if the regulator appears positively,
      - '0' if it appears as Not(regulator),
      - '-' if it does not appear.
    """
    cover = {reg.name: '-' for reg in regulators}
    
    # If term is an AND, break into literals; otherwise, treat it as a single literal.
    if isinstance(term, And):
        literals = term.args
    else:
        literals = [term]
    
    for lit in literals:
        if isinstance(lit, Not):
            var = lit.args[0]
            cover[var.name] = '0'
        else:
            cover[lit.name] = '1'
    
    return "".join(cover[reg.name] for reg in regulators)

def process_node(node_name, expr_str, global_symbols):
    """
    Process a single node’s Boolean expression.
    Returns a tuple of:
      - regulators: the list of regulator symbols (ordered by global index)
      - covers: a list of cover strings (each with " 1" appended)
    """
    # Build a local namespace mapping each node to its sympy symbol
    local_dict = {name: global_symbols[name]['symbol'] for name in global_symbols}
    try:
        expr = sympify(expr_str, locals=local_dict)
    except Exception as e:
        raise ValueError(f"Error parsing expression for node {node_name}: {expr_str}") from e

    # Simplify expression to disjunctive normal form (DNF)
    dnf_expr = simplify_logic(expr, form='dnf')
    regulators = get_regulators(expr, global_symbols)
    
    covers = []
    if dnf_expr == BooleanTrue:
        # Constant true
        if regulators:
            covers.append("-" * len(regulators) + " 1")
        else:
            covers.append("1")
    elif dnf_expr == BooleanFalse:
        # Constant false (rare in typical attractor models)
        if regulators:
            covers.append("-" * len(regulators) + " 0")
        else:
            covers.append("0")
    else:
        # dnf_expr may be an OR of terms or a single term.
        if isinstance(dnf_expr, Or):
            terms = dnf_expr.args
        else:
            terms = [dnf_expr]
        for term in terms:
            cover_str = term_to_cover(term, regulators)
            covers.append(cover_str + " 1")
    return regulators, covers

def convert_bnet_to_thanalia(input_file, output_file):
    # Parse the input .bnet file.
    nodes_data = parse_bnet_file(input_file)
    # Build a global mapping: each node gets an index and a sympy symbol.
    global_symbols = {}
    for idx, (node, _) in enumerate(nodes_data, start=1):
        global_symbols[node] = {'index': idx, 'symbol': symbols(node)}
    
    with open(output_file, 'w') as fout:
        # Write header lines.
        fout.write("# Boolean network model converted from .bnet format\n")
        fout.write("# Converted by convert_bnet_to_thanalia.py script\n\n")
        
        # Write the total number of nodes.
        num_nodes = len(nodes_data)
        fout.write(".v {}\n\n".format(num_nodes))
        
        # Write labels: each node gets a line like ".l index nodename"
        for node, _ in nodes_data:
            idx = global_symbols[node]['index']
            fout.write(".l {} {}\n".format(idx, node))
        fout.write("\n")
        
        # Process each node and write its block.
        for node, expr_str in nodes_data:
            idx = global_symbols[node]['index']
            fout.write("# {} = {}\n".format(idx, node))
            regulators, covers = process_node(node, expr_str, global_symbols)
            if regulators:
                # Write number of regulators and list of regulator indices.
                reg_indices = [str(global_symbols[reg.name]['index']) for reg in regulators]
                fout.write(".n {} {} {}\n".format(idx, len(regulators), " ".join(reg_indices)))
            else:
                fout.write(".n {} 0\n".format(idx))
            for cover_line in covers:
                fout.write(cover_line + "\n")
            fout.write("\n")
        
        # End-of-file marker.
        fout.write(".e end of file\n")
    print("Conversion completed. Output saved to '{}'.".format(output_file))

def main():
    parser = argparse.ArgumentParser
