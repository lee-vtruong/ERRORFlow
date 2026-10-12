# FEVER V0/V1 archive report

## Outcome

The pre-V2 FEVER implementation has been archived without copying large server
artifacts or changing user-owned work. Its code boundary is commit
`a4280b8ac550909beb57b5509e10bf71cd289304`; practical source anchors are in
`../manifests/checksums.sha256`.

The local compatibility test passed: **32 tests passed**. This verifies the
checked-out implementation, not the continued availability or byte identity of
server-only data and model files.

Run `python scripts/verify_fever_v1_archive.py` from any descendant revision to
verify checksum anchors against the audited commit. Direct comparison with V2
working-tree files is expected to differ where migration changed those files.

## Historical result boundary

V0 completed on all 19,998 FEVER shared-task development examples. V1 completed
on the 1,000-example slice. Full V1 candidate generation had begun and a full
title-candidate file was reported, but no completed full V1 end-to-end score was
available at archive time. The archive therefore does not manufacture one.

## Rollback

1. Preserve or commit any current work; never reset a dirty tree blindly.
2. Create a separate worktree from the audited commit:

   ```bash
   git worktree add ../ERRORFlow-fever-v1 a4280b8ac550909beb57b5509e10bf71cd289304
   ```

3. Activate the historical environment or recreate one from the environment
   snapshot.
4. Verify the server-only paths listed in `files_manifest.json`.
5. Run `python -m pytest -q` and require the legacy suite to pass before jobs.
6. Replay commands from `original_commands.md` into a new output directory.

After the current dirty documentation is reviewed and committed, an optional
annotated tag such as `fever-v1-pre-v2` may be created on the intended commit.
The audit deliberately did not create that tag automatically.
