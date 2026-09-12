# Real-browser acceptance checklist — pending

These checks were NOT executed in a real browser during this delivery.

1. Open HTML offline. Check console for errors and layout at desktop and mobile widths.
2. Open included test-video.mp4. Play, pause, seek, change speed and step using configured 25 FPS.
3. Add a manual observation. Place all four points, draw a boundary and verify signed pixel distances.
4. Flip the boundary, mark a tyre missing, then undo/redo. Verify audit records accumulate.
5. Toggle each quality flag. Measurement must become unavailable, not zero.
6. Switch timestamps away from the selected observation. Overlay must disappear; jump back to restore it.
7. Select crop from the original player while in edit mode. Crop must receive pointer input.
8. In a new clip/session, load simulated predictions. Confirm the header, video and workbench disclose simulation.
9. Try malformed JSON, duplicate IDs, wrong video dimensions, reversed intervals and scores above one. Import must fail without changing prior data.
10. Create a manual interval, jump to it, remove it and undo the removal.
11. Save a human assessment and export JSON. Inspect original points, corrected points, flags, audit and notes separately.
12. Refresh and reopen the same video. Saved edits should return; in-memory undo stacks should be empty.
13. Select multiple clips and switch rapidly. No overlay or edit may be applied to a different clip.
14. Test Ctrl/Cmd+Z and N outside text inputs. Typing in notes must not trigger navigation.
15. Fill browser storage with a large test set only in a disposable profile. Saving must warn; JSON export must remain available.
