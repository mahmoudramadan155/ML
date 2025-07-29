from flask import Flask, render_template
from flask import make_response, request, abort
import traceback

# import all the app logic
from analysis import get_logo

# Initialize Flask web application
app = Flask(__name__)


@app.route('/')
def index():
    """ This is our home page """
    name = request.args.get('name')
    return render_template('index.html', name=name)


@app.route('/logo/<name>.png')
def logo(name):
    """
    Gene name parameter is passed along as part of the image name requested
    Generate logo as an image
    """
    print(f"\n=== Logo request for gene: {name} ===")
    
    if len(name) == 0:
        print("Empty gene name provided")
        abort(404)

    logo_data = None
    try:
        print(f"Calling get_logo for: {name}")
        logo_data = get_logo(name)
        
        if logo_data is None:
            print(f"get_logo returned None for gene: {name}")
            abort(404)
            
        print(f"Logo generation successful, data size: {len(logo_data)} bytes")
        
    except Exception as e:
        print(f"Exception in logo generation for {name}: {e}")
        traceback.print_exc()
        abort(500)

    response = make_response(logo_data)
    response.headers['Content-Type'] = 'image/png'
    return response


@app.route('/test/<name>')
def test_gene(name):
    """
    Test endpoint to check gene processing without image generation
    """
    try:
        from analysis import get_variants, get_genome_sequence_ensembl
        
        print(f"Testing gene: {name}")
        variants = get_variants(name)
        
        if not variants:
            return f"No variants found for gene: {name}"
            
        chrom = list(variants.keys())[0][0]
        positions = [k[1] for k in variants.keys()]
        start = min(positions)
        end = max(positions)
        
        sequence = get_genome_sequence_ensembl(chrom, start, end)
        
        result = f"""
        Gene: {name}
        Variants found: {len(variants)}
        Region: {chrom}:{start}-{end}
        Sequence length: {len(sequence) if sequence else 'None'}
        Sample variants: {dict(list(variants.items())[:3])}
        """
        
        return f"<pre>{result}</pre>"
        
    except Exception as e:
        return f"Error testing gene {name}: {str(e)}<br><pre>{traceback.format_exc()}</pre>"


if __name__ == '__main__':
    """ 
      Application entry point
      This web application has a built-in web server 
    """
    print("Starting Flask app in debug mode...")
    app.run(debug=True)
