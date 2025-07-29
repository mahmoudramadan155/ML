"""
This module contains application logic implementation:

  1. Fetching data from UCSC genomes
  2. Creating image with genetic variants for a given gene

Note that it requires ghostscript for image generation and pip install weblogo pymysql sqlalchemy
"""

from sqlalchemy import *
from sqlalchemy.orm import registry, Session
from weblogo import *
from io import StringIO
import requests
import traceback

#########
# Initialize global variable with connection to database
engine = create_engine('mysql+pymysql://genome@genome-mysql.cse.ucsc.edu/hg38')

###########
# Get database tables metadata from the database - Updated for SQLAlchemy 2.0+
meta = MetaData()

try:
    meta.reflect(bind=engine, only=['refGene', 'snp147Common'])
    print("✓ Database connection and reflection successful")
except Exception as e:
    print(f"✗ Database reflection failed: {e}")
    raise

gene_table = Table('refGene',
    meta,
    PrimaryKeyConstraint('name'),
    extend_existing=True)

snp_table = Table('snp147Common',
    meta,
    PrimaryKeyConstraint('name'),
    extend_existing=True)

##################
# ORM mapping of database tables to Python objects using SQLAlchemy 2.0+ registry
mapper_registry = registry()

class DatabaseObject(object):
    def __repr__(self):
        return "\n".join(
            ["{:20s}: {}".format(key, self.__dict__[key]) for key in sorted(self.__dict__.keys())]
        )

class Gene(DatabaseObject):
    def __repr__(self):
        return("Gene {} ({})\nCDS location {} {}-{} on strand {}".format(
            self.name, self.name2, self.chrom, self.cdsStart, self.cdsEnd, self.strand))

class SNP(DatabaseObject):
    snp_class = Column('class', String)

# Create mapping between Classes and tables using new SQLAlchemy 2.0+ syntax
try:
    mapper_registry.map_imperatively(SNP, snp_table)
    mapper_registry.map_imperatively(Gene, gene_table)
    print("✓ ORM mapping successful")
except Exception as e:
    print(f"✗ ORM mapping failed: {e}")
    raise

# Create session
session = Session(engine)

def get_genome_sequence_ensembl(chrom, start, end):
    """
    Auxiliary function
    API described here http://rest.ensembl.org/documentation/info/sequence_region
    
    returns DNA sequence for a given human genome location
    """
    try:
        url = 'https://rest.ensembl.org/sequence/region/human/{0}:{1}..{2}:1?content-type=application/json'.format(chrom, start, end)
        print(f"Fetching sequence from: {url}")
        r = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10.000)
        if not r.ok:
            print("REST Request FAILED")
            decoded = r.json()
            print(decoded['error'])
            return None
        else:
            print("REST Request OK")
            decoded = r.json()
            sequence = decoded['seq']
            print(f"Retrieved sequence length: {len(sequence)}")
            return sequence
    except Exception as e:
        print(f"Error in get_genome_sequence_ensembl: {e}")
        traceback.print_exc()
        return None

def get_variants(gene):
    """
       Auxiliary function
       Returns SNPs - variants for a given gene by querying the database
    """
    try:
        print(f"Searching for gene: {gene}")
        variants = {}
        genes_found = session.query(Gene).filter(Gene.name2 == gene).filter(Gene.cdsEnd > Gene.cdsStart).all()
        
        if not genes_found:
            print(f"No genes found for: {gene}")
            return variants
            
        print(f"Found {len(genes_found)} gene records for {gene}")
        
        for g in genes_found:
            print(f"Processing gene record: {g.name} on {g.chrom}:{g.cdsStart}-{g.cdsEnd}")
            
            snps = session.query(SNP).filter(
                SNP.snp_class == 'single').filter(
                SNP.strand == g.strand).filter(
                SNP.chrom == g.chrom).filter(
                SNP.chromStart >= g.cdsStart).filter(
                SNP.chromEnd <= g.cdsEnd).all()
                
            print(f"Found {len(snps)} SNPs for this gene")
            
            for s in snps:
                try:
                    alleles = s.alleles.decode('utf-8')[:-1].split(",")
                    variants[(s.chrom, s.chromStart)] = alleles
                except Exception as e:
                    print(f"Error processing SNP {s.name}: {e}")
                    continue
                    
            break  # analyze only one gene record, skip the rest - for testing
            
        print(f"Total variants collected: {len(variants)}")
        return variants
        
    except Exception as e:
        print(f"Error in get_variants: {e}")
        traceback.print_exc()
        return {}

def get_logo(gene):
    """
       Generates logo image using weblogo library. Input is gene name. Output is PNG image.
       This function retrives all gene variants, retrieves the sequence
       Maps variants onto sequence
    """
    
    try:
        print(f"=== Starting logo generation for gene: {gene} ===")
        
        # this is a limit on sequence length for practical reasons
        SEQUENCE_LENGTH_LIMIT = 1000

        # retrieve SNPs
        variants = get_variants(gene)
        
        if not variants:
            print("No variants found, cannot generate logo")
            return None

        chrom = list(variants.keys())[0][0]
        positions = [k[1] for k in variants.keys()]
        start = min(positions)
        end = max(positions)
        
        print(f"Sequence region: {chrom}:{start}-{end}")

        # retrieve sequence
        sequence = get_genome_sequence_ensembl(chrom, start, end)
        
        if not sequence:
            print("Failed to retrieve sequence")
            return None

        print(f"Original sequence length: {len(sequence)}")
        if len(sequence) > SEQUENCE_LENGTH_LIMIT:
            sequence = sequence[:SEQUENCE_LENGTH_LIMIT]
            print(f"Truncated sequence length: {len(sequence)}")

        # Build sequence data for weblogo
        seqs = ">\n"+sequence+"\n"

        variant_count = 0
        for (chrom, position), alleles in variants.items():
            if position - start >= len(sequence):
                continue
            
            for allele in alleles:
                seq = ["-"] * len(sequence)
                seq[position-start] = allele
                seqs += ">\n" + "".join(seq) + "\n"
                variant_count += 1

        print(f"Added {variant_count} variant sequences")
        print(f"Total sequence data length: {len(seqs)} characters")

        # Generate logo using weblogo
        print("Parsing sequences...")
        sequences = read_seq_data(StringIO(seqs))
        
        print("Creating logo data...")
        data = LogoData.from_seqs(sequences)
        
        print("Setting up options...")
        options = LogoOptions()
        options.title = f'Gene {gene} Variants'
        options.scale_width = False
        options.logo_end = SEQUENCE_LENGTH_LIMIT
        options.stacks_per_line = 50
        
        print("Creating formatting...")
        formatting = LogoFormat(data, options)

        print("Generating PNG...")
        png = png_formatter(data, formatting)
        
        print(f"✓ Successfully generated logo for {gene}, PNG size: {len(png)} bytes")
        return png
        
    except Exception as e:
        print(f"✗ Error in get_logo for gene {gene}: {e}")
        traceback.print_exc()
        return None
    