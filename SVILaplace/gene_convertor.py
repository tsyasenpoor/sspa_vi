import json
import os
import re
import mygene

class GeneIDConverter:
    """
    A class to handle gene ID conversions with persistent caching.
    Caches conversions between gene symbols and Ensembl IDs to avoid repeated API calls.
    """
    
    def __init__(self, cache_file='/labs/Aguiar/SSPA_BRAY/BRay/gene_id_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.mg = mygene.MyGeneInfo()
        
    def _load_cache(self):
        """Load existing cache from file, or create empty cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                print(f"Loaded gene ID cache with {len(cache.get('symbol_to_ensembl', {}))} symbol mappings")
                return cache
            except Exception as e:
                print(f"Error loading cache: {e}. Creating new cache.")
                return {'symbol_to_ensembl': {}, 'ensembl_to_symbol': {}}
        else:
            return {'symbol_to_ensembl': {}, 'ensembl_to_symbol': {}}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _is_ensembl_id(self, gene_id):
        """
        Check if a gene ID is in Ensembl format.
        Ensembl mouse IDs start with 'ENSMUSG', human with 'ENSG'
        """
        if not isinstance(gene_id, str):
            return False
        return bool(re.match(r'^ENS[A-Z]*G\d{11}$', gene_id))
    
    def _detect_gene_format(self, gene_list):
        """
        Detect if gene list is primarily Ensembl IDs or gene symbols.
        Returns: 'ensembl', 'symbol', or 'mixed'
        """
        if not gene_list:
            return 'unknown'
        
        sample_size = min(100, len(gene_list))
        sample = gene_list[:sample_size]
        
        ensembl_count = sum(1 for g in sample if self._is_ensembl_id(g))
        ensembl_ratio = ensembl_count / sample_size
        
        if ensembl_ratio > 0.9:
            return 'ensembl'
        elif ensembl_ratio < 0.1:
            return 'symbol'
        else:
            return 'mixed'
    
    def symbols_to_ensembl(self, gene_symbols, species='mouse', force_update=False):
        """
        Convert gene symbols to Ensembl IDs.
        Automatically detects if input is already in Ensembl format.
        
        Parameters:
        -----------
        gene_symbols : list
            List of gene symbols or Ensembl IDs to convert
        species : str
            Species name (default: 'mouse')
        force_update : bool
            If True, bypass cache and force API call
            
        Returns:
        --------
        dict : mapping of input -> Ensembl ID (or input if already Ensembl)
        list : list of Ensembl IDs in same order as input
        """
        # Detect format
        gene_format = self._detect_gene_format(gene_symbols)
        
        if gene_format == 'ensembl':
            print(f"Detected {len(gene_symbols)} genes already in Ensembl format - skipping conversion")
            # Return identity mapping
            results_map = {gene: gene for gene in gene_symbols}
            return results_map, gene_symbols
        
        print(f"Detected gene format: {gene_format}")
        
        # Proceed with conversion for symbols and mixed formats
        genes_to_query = []
        results_map = {}
        
        # Separate already-Ensembl IDs from symbols
        for gene in gene_symbols:
            if self._is_ensembl_id(gene):
                # Already Ensembl, keep as-is
                results_map[gene] = gene
            elif not force_update and gene in self.cache['symbol_to_ensembl']:
                # In cache
                results_map[gene] = self.cache['symbol_to_ensembl'][gene]
            else:
                # Need to query
                genes_to_query.append(gene)
        
        # Query API for uncached genes
        if genes_to_query:
            print(f"Querying {len(genes_to_query)} genes from MyGene API...")
            query_results = self.mg.querymany(genes_to_query, scopes='symbol', 
                                             fields='ensembl.gene', species=species)
            
            # Process and cache results
            for item in query_results:
                symbol = item['query']
                if 'ensembl' in item and item['ensembl']:
                    if isinstance(item['ensembl'], list):
                        ensembl_id = item['ensembl'][0]['gene']
                    else:
                        ensembl_id = item['ensembl']['gene']
                    
                    # Update cache
                    self.cache['symbol_to_ensembl'][symbol] = ensembl_id
                    self.cache['ensembl_to_symbol'][ensembl_id] = symbol
                    results_map[symbol] = ensembl_id
                else:
                    # Store None for genes without Ensembl ID
                    self.cache['symbol_to_ensembl'][symbol] = None
                    results_map[symbol] = None
            
            # Save updated cache
            self._save_cache()
            print(f"Added {len(genes_to_query)} new genes to cache")
        else:
            print(f"All {len(gene_symbols)} genes found in cache or already in Ensembl format")
        
        # Return results in original order
        ensembl_ids = [results_map.get(gene) for gene in gene_symbols]
        return results_map, ensembl_ids
    
    def ensembl_to_symbols(self, ensembl_ids, species='mouse', force_update=False):
        """
        Convert Ensembl IDs to gene symbols.
        Automatically detects if input is already in symbol format.
        
        Parameters:
        -----------
        ensembl_ids : list
            List of Ensembl IDs or gene symbols to convert
        species : str
            Species name (default: 'mouse')
        force_update : bool
            If True, bypass cache and force API call
            
        Returns:
        --------
        dict : mapping of Ensembl ID -> symbol (or input if already symbol)
        list : list of symbols in same order as input
        """
        # Detect format
        gene_format = self._detect_gene_format(ensembl_ids)
        
        if gene_format == 'symbol':
            print(f"Detected {len(ensembl_ids)} genes already in symbol format - skipping conversion")
            # Return identity mapping
            results_map = {gene: gene for gene in ensembl_ids}
            return results_map, ensembl_ids
        
        print(f"Detected gene format: {gene_format}")
        
        # Proceed with conversion
        ids_to_query = []
        results_map = {}
        
        # Separate already-symbols from Ensembl IDs
        for gene in ensembl_ids:
            if not self._is_ensembl_id(gene):
                # Already a symbol, keep as-is
                results_map[gene] = gene
            elif not force_update and gene in self.cache['ensembl_to_symbol']:
                # In cache
                results_map[gene] = self.cache['ensembl_to_symbol'][gene]
            else:
                # Need to query
                ids_to_query.append(gene)
        
        # Query API for uncached IDs
        if ids_to_query:
            print(f"Querying {len(ids_to_query)} Ensembl IDs from MyGene API...")
            query_results = self.mg.querymany(ids_to_query, scopes='ensembl.gene', 
                                             fields='symbol', species=species)
            
            # Process and cache results
            for item in query_results:
                ensembl_id = item['query']
                if 'symbol' in item:
                    symbol = item['symbol']
                    
                    # Update cache
                    self.cache['ensembl_to_symbol'][ensembl_id] = symbol
                    self.cache['symbol_to_ensembl'][symbol] = ensembl_id
                    results_map[ensembl_id] = symbol
                else:
                    # Store None for IDs without symbol
                    self.cache['ensembl_to_symbol'][ensembl_id] = None
                    results_map[ensembl_id] = None
            
            # Save updated cache
            self._save_cache()
            print(f"Added {len(ids_to_query)} new IDs to cache")
        else:
            print(f"All {len(ensembl_ids)} genes found in cache or already in symbol format")
        
        # Return results in original order
        symbols = [results_map.get(eid) for eid in ensembl_ids]
        return results_map, symbols
    
    def get_cache_stats(self):
        """Print statistics about the cache"""
        print(f"Cache statistics:")
        print(f"  Total symbol->Ensembl mappings: {len(self.cache['symbol_to_ensembl'])}")
        print(f"  Total Ensembl->symbol mappings: {len(self.cache['ensembl_to_symbol'])}")
        print(f"  Cache file: {self.cache_file}")