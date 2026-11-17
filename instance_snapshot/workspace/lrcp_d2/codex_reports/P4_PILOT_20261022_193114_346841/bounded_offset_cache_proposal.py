"""PROPOSED next-run CPU memory fix. Not installed in the stopped runner.
No inference launcher. Existing dataset source is never edited.
"""
class BoundedOffsetDataset:
    """Delegate native __getitem__ once, then detach ONLY retained offset storage.

    Native item contents, order and RNG remain unchanged. The current item's
    old view may live until collation; the persistent scene cache stores the
    exact small tensor instead of retaining its complete multi-agent NPY.
    Full-history equivalence still requires the next run's original gates.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.compacted_count = 0
        self.released_storage_bytes = 0

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        scene = self.dataset.scene_database[index]
        for key, entry in scene.items():
            if key == 'prev':
                continue
            params = entry.get('params', {})
            offset = params.get('offset')
            if offset is None:
                continue
            size = offset.numel() * offset.element_size()
            storage_bytes = offset.untyped_storage().nbytes()
            if storage_bytes <= size:
                continue
            detached = offset.clone()
            if detached.shape != offset.shape or detached.dtype != offset.dtype or detached.device != offset.device:
                raise RuntimeError('OFFSET_COMPACTION_METADATA_DIFFERENCE')
            if not detached.equal(offset):
                raise RuntimeError('OFFSET_COMPACTION_VALUE_DIFFERENCE')
            if detached.untyped_storage().nbytes() != size:
                raise RuntimeError('OFFSET_COMPACTION_NOT_BOUNDED')
            params['offset'] = detached
            self.compacted_count += 1
            self.released_storage_bytes += storage_bytes - size
        return item
