#!/usr/bin/env python3

import os
import asyncio
import aiohttp
import json_stream
import json_stream.dump
import bz2
import json
from fastembed import TextEmbedding
import argparse

from trustgraph.knowledge import Uri, Literal
from trustgraph.knowledge import IS_A, LABEL, NAME, DESCRIPTION, URL
from trustgraph.knowledge import hash



def prop_uri(type):
    return Uri(f"https://trustgraph.ai/e/hetionet/prop/{type}")

def node_type_uri(type):
    return Uri(f"https://trustgraph.ai/e/hetionet/node-type/{type}")

def edge_uri(type):
    return Uri(f"https://trustgraph.ai/e/hetionet/edge/{type}")

def node_uri(type, id):
    id = hash(f"{type}/{id}")
    return Uri(f"https://trustgraph.ai/e/hetionet/node/{id}")

class Processor:

    def __init__(
            self, input_file, url, user, collection, verbose,
            embeddings_model, doc_id
    ):

        self.verbose = verbose

        if self.verbose:
            print("Initialise...")

        self.path = input_file

        self.user = user
        self.collection = collection

        self.doc_id = doc_id
        self.embeddings = TextEmbedding(model_name = embeddings_model)

        self.url = url

        if self.url[-1] != '/':
            self.url += '/'

        if self.verbose:
            print("Initialised.")

    async def process_ec(self, ent, context):

        vecs = self.embeddings.embed([context])
        vecs = [v.tolist() for v in vecs]

        t = {
            "metadata": {
                "id": self.doc_id,
                "metadata": [],
                "user": self.user,
                "collection": self.collection
            },
            "entities": [
                {
                    "entity": {
                        "v": ent, "e": isinstance(ent, Uri)
                    },
                    "vectors": vecs
                }
            ]
        }

        await self.ge_q.put(t)

    async def process_triple(self, t):

        t = {
            "metadata": {
                "id": self.doc_id,
                "metadata": [],
                "user": self.user,
                "collection": self.collection
            },
            "triples": [
                {
                    "s": {
                        "v": t[0], "e": isinstance(t[0], Uri)
                    },
                    "p": {
                        "v": t[1], "e": isinstance(t[1], Uri)
                    },
                    "o": {
                        "v": t[2], "e": isinstance(t[2], Uri)
                    }
                }
            ]
        }

        await self.tr_q.put(t)

    async def process_node_type(self, type):

        if self.verbose:
            print("Node type:", type)

        await self.process_triple((
            node_type_uri(type), Uri(LABEL), Literal(type)
        ))

        await self.process_ec(node_type_uri(type), type)

    async def process_edge_type(self, edge_type):

        source_type, target_type, type, direction = edge_type

        if self.verbose:
            print("Edge type:", type)

        await self.process_triple((
            edge_uri(type), Uri(LABEL), Literal(type)
        ))

        await self.process_ec(edge_uri(type), type)

    async def process_node(self, node):

        type = node["kind"]
        identifier = node["identifier"]
        name = node["name"]
        data = node["data"]

        if self.verbose:
            print("Node:", type, identifier)

        await self.process_triple((
            node_uri(type, identifier), Uri(IS_A), node_type_uri(type)
        ))
        await self.process_triple((
            node_uri(type, identifier), Uri(LABEL), Literal(name)
        ))

        # Available data fields:
        #   bto_id
        #   chromosome
        #   class_type
        #   description
        #   inchi
        #   inchikey
        #   license
        #   mesh_id
        #   source
        #   url

        for k, v in data.items():

            if k == "description":
                await self.process_triple((
                    node_uri(type, identifier), Uri(DESCRIPTION), Literal(v)
                ))
                await self.process_ec(node_uri(type, identifier), v)

            if k == "chromosome":
                await self.process_triple((
                    node_uri(type, identifier), prop_uri("chromosome"),
                    Literal(v)
                ))

            if k == "source":
                await self.process_triple((
                    node_uri(type, identifier), prop_uri("source"), Literal(v)
                ))
                await self.process_ec(node_uri(type, identifier), v)

            if k == "sources":
                for w in v:
                    await self.process_triple((
                        node_uri(type, identifier), prop_uri("source"),
                        Literal(w)
                    ))
                    await self.process_ec(node_uri(type, identifier), v)

    async def process_edge(self, edge):

        source_type, source_id = edge["source_id"]
        target_type, target_id = edge["target_id"]
        type = edge["kind"]
        direction = edge["direction"]
        data = edge["data"]

        if self.verbose:
            print("Edge:", source_type, source_id, target_type, target_id)

        await self.process_triple((
            node_uri(source_type, source_id),
            edge_uri(type),
            node_uri(target_type, target_id)
        ))

        # Available data fields:
        #   actions
        #   affinity_nM
        #   license
        #   log2_fold_change
        #   method
        #   pubmed_ids
        #   similarity
        #   source
        #   sources
        #   subtypes
        #   unbiased
        #   url
        #   urls
        #   z_score

    async def process(self):

        await self.process_triple((
            Uri(self.doc_id), Uri(LABEL), Literal("Hetionet")
        ))

        await self.process_triple((
            Uri(DESCRIPTION), Uri(LABEL), Literal("description")
        ))
        await self.process_triple((
            prop_uri("chromosome"), Uri(LABEL), Literal("chromosome")
        ))
        await self.process_triple((
            prop_uri("source"), Uri(LABEL), Literal("source")
        ))

        with bz2.open(self.path, "r") as f:

            stream = json_stream.load(f)

            mnk = stream["metanode_kinds"]
            for v in mnk:
                await self.process_node_type(v)

            mnt = stream["metaedge_tuples"]
            for v in mnt:
                await self.process_edge_type(v)

            nodes = stream["nodes"]
            for v in nodes:
                await self.process_node(v)

            edges = stream["edges"]
            for v in edges:
                await self.process_edge(v)

        await self.ge_q.put(None)
        await self.tr_q.put(None)

    async def queue_runner(self, q, url):

        if self.verbose:
            print("Queue runner for", url, "starting...")

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:

                while True:

                    item = await q.get()

                    if item is None: break

                    await ws.send_json(item)

                if self.verbose:
                    print("Processing complete for", url)

    async def run(self):

        url = self.url + "api/v1/"

        self.running = True

        self.ge_q = asyncio.Queue(maxsize=1000)
        self.tr_q = asyncio.Queue(maxsize=1000)

        self.ge_task = asyncio.create_task(
            self.queue_runner(self.ge_q, f"{url}load/graph-embeddings")
        )

        self.tr_task = asyncio.create_task(
            self.queue_runner(self.tr_q, f"{url}load/triples")
        )

        self.process_task = asyncio.create_task(
            self.process()
        )

        await asyncio.gather(self.ge_task, self.tr_task, self.process_task)

        print("All complete.")

def run():
    
    parser = argparse.ArgumentParser(
        prog='tg-load-kg-core',
        description=__doc__,
    )

    default_url = os.getenv("TRUSTGRAPH_API", "http://localhost:8088")
    default_user = "trustgraph"
    default_collection = "default"
    default_model = "sentence-transformers/all-MiniLM-L6-v2"
    default_doc_id = "https://trustgraph/e/hetionet"

    parser.add_argument(
        '-u', '--url',
        default=default_url,
        help=f'TrustGraph API URL (default: {default_url})',
    )

    parser.add_argument(
        '-i', '--input-file',
        # Make it mandatory, difficult to over-write an existing file
        required=True,
        help=f'Output file'
    )

    parser.add_argument(
        '--user',
        default=default_user,
        help=f'User ID to load as (default: {default_user})'
    )

    parser.add_argument(
        '--collection',
        default=default_collection,
        help=f'Collection ID to load as (default: {default_collection})'
    )

    parser.add_argument(
        '--embeddings-model',
        default=default_model,
        help=f'User ID to load as (default: {default_model})'
    )

    parser.add_argument(
        '--doc-id',
        default=default_doc_id,
        help=f'User ID to load as (default: {default_doc_id})'
    )

    parser.add_argument(
        '-v', '--verbose',
        action="store_true",
        help=f'Verbose output'
    )

    args = parser.parse_args()
    args = vars(args)

    p = Processor(**args)
    asyncio.run(p.run())

run()