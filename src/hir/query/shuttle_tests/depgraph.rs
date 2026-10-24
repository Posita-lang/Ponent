use super::super::dep_graph::{ContentClass, DepGraph, Durability, Quiesced};
use super::*;
use std::sync::Arc;

/// Test-only quiescence proof for direct dep-graph calls (no QuerySystem
/// here) — same escape hatch as dep_graph's own unit tests.
fn q() -> Quiesced {
    Quiesced::assume_for_test()
}

#[test]
fn dur_prop_late_recheck_catches_mid_compute_lowering() {
    run_check(|| {
        let sched = TestScheduler::new();
        let dg: &'static DepGraph<TestScheduler> =
            Box::leak(Box::new(DepGraph::new(Arc::new(sched.clone()))));
        let m = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(m, Durability::HIGH);
        let x = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(x, Durability::HIGH);

        sched.arm(move |p| matches!(p, SchedulePoint::FinishWindow { node } if *node == x.0));
        let h = shuttle::thread::spawn(move || {
            dg.start_task(x);
            dg.read(m);
            dg.finish_task_and_publish(x, Some(1))
        });
        sched.wait_parked();
        assert!(
            dg.lower_and_propagate(m, Durability::LOW),
            "the lowering must apply while the finish is parked"
        );
        assert_eq!(
            dg.durability_of(x),
            Durability::HIGH,
            "the cascade must miss the un-flushed reader"
        );
        sched.release_all();
        let changed = h.join().expect("finish thread must not panic");
        assert!(!changed, "M's level drop never changed X's fingerprint");
        assert_eq!(dg.durability_of(m), Durability::LOW);
        assert_eq!(
            dg.durability_of(x),
            Durability::LOW,
            "the late re-aggregation must catch the mid-compute lowering"
        );
    });
}

#[test]
fn dur_prop_seqlock_seals_green_verdict_against_inflight_lowering() {
    run_check(|| {
        let sched = TestScheduler::new();
        let dg: &'static DepGraph<TestScheduler> =
            Box::leak(Box::new(DepGraph::new(Arc::new(sched.clone()))));
        let x = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(x, Durability::HIGH);
        dg.start_task(x);
        dg.finish_task();
        assert!(!dg.finish_task_and_publish(x, Some(1)));
        let warm = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(warm, Durability::HIGH);
        dg.invalidate(warm, &q());
        assert_eq!(dg.verify_green_transitive_with_dirty(x), None);
        let vl_before = dg.verified_lvl_of(x);
        assert!(vl_before > 0);
        assert_eq!(dg.durability_of(x), Durability::HIGH);

        sched.arm(move |p| matches!(p, SchedulePoint::VerifyWindow { node } if *node == x.0));
        let h = shuttle::thread::spawn(move || dg.verify_green_transitive_with_dirty(x));
        sched.wait_parked();
        assert!(
            dg.lower_and_propagate(x, Durability::LOW),
            "the bracketing lowering must apply while the verdict is parked"
        );
        sched.release_then_arm(
            move |p| matches!(p, SchedulePoint::VerifyWindow { node } if *node == x.0),
        );
        sched.wait_parked();
        sched.release_all();
        assert_eq!(
            h.join().expect("verify thread must not panic"),
            None,
            "the retry's verdict is still green (a lowering never marks)"
        );
        assert_eq!(dg.durability_of(x), Durability::LOW);
        let low = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.invalidate(low, &q());
        assert_eq!(dg.verify_green_transitive_with_dirty(x), None);
        assert!(
            dg.verified_lvl_of(x) > vl_before,
            "the post-lowering re-stamp must ride the LOW counter, not the abandoned HIGH one"
        );
    });
}

/// The publish-time fan-out is LOCK-FREE, so a mark of a
/// FINISHING node can land in the window between that node's in-section
/// `clear_dirty` and its post-flush late re-check.
///
/// Two phases, one per half of the chain that repairs such a mark, because
/// neither half covers the other's case:
///
///  * Phase A - the finishing node ALREADY has a reverse edge on its read,
///    so R's fan-out reaches it directly.
///  * Phase B - the finishing node reads R for the FIRST time, so it has NO
///    reverse edge yet (that is flushed by its own finish, after the park).
///    R's fan-out therefore CANNOT reach it, and only the late re-check can
///    restore the bit.
///
/// Both phases park the owner at `FinishWindow`, which the finish path emits
/// AFTER its in-section colour decision and BEFORE its late re-check - i.e.
/// inside the window - and republish R while it sits there.
///
/// Phase B is the load-bearing one: it is the only place the LATE RE-CHECK is
/// the sole repair, and it goes straight to the erasure in question
/// about.  (The micro-window inside the shard-write closure, between the
/// suspect evaluation and `clear_dirty` itself, has no suspension point and is
/// therefore not separately targetable - see the note at the end.)
///
/// PUBLISH-WINDOW FAMILY (grep `publish_window` and `FinishWindow`): this is
/// the exhaustive-interleaving member.  The unit-level member is
/// `dep_graph::tests::fanout_marks_every_reader_of_a_popular_node` (the
/// fan-out mechanism itself, 64 readers) and the end-to-end OS-thread member
/// is `defense_tests::publish_window_suspect_value_never_served_green` (the
/// getter-side symptom a reader would actually observe).  The three are
/// deliberately not redundant: this one proves the REPAIR, the unit one
/// proves the fan-out, and the defense one proves the symptom is unobservable.
#[test]
fn finish_window_erased_fanout_mark_is_repaired() {
    run_check(|| {
        let sched = TestScheduler::new();
        let dg: &'static DepGraph<TestScheduler> =
            Box::leak(Box::new(DepGraph::new(Arc::new(sched.clone()))));
        let r = dg.allocate_node_index(ContentClass::SEMANTIC);
        let n_fanned = dg.allocate_node_index(ContentClass::SEMANTIC);
        let n_late = dg.allocate_node_index(ContentClass::SEMANTIC);

        // R must already own a fingerprint so its later republish counts as a
        // CHANGE - the fan-out only runs in the `if changed` arm.  This FIRST
        // publish is not itself a change (there was no prior fingerprint to
        // differ from), so it returns false.
        dg.start_task(r);
        assert!(!dg.finish_task_and_publish(r, Some(1)));
        // n_fanned reads R, so its reverse edge on R exists and a republish
        // of R can actually reach it.  n_late does NOT read R at all here.
        dg.start_task(n_fanned);
        dg.read(r);
        assert!(!dg.finish_task_and_publish(n_fanned, Some(1)));
        dg.start_task(n_late);
        assert!(!dg.finish_task_and_publish(n_late, Some(1)));
        assert!(
            dg.verify_green_transitive_with_dirty(n_fanned).is_none()
                && dg.verify_green_transitive_with_dirty(n_late).is_none(),
            "both finishing nodes must start green for the interleave to be meaningful"
        );

        // -- Phase A: the fan-out reaches the parked finisher directly --
        sched.arm(move |p| matches!(p, SchedulePoint::FinishWindow { node } if *node == n_fanned.0));
        let ha = shuttle::thread::spawn(move || {
            dg.start_task(n_fanned);
            dg.read(r);
            dg.finish_task_and_publish(n_fanned, Some(2))
        });
        sched.wait_parked();
        // R republishes with a CHANGED fingerprint: its lock-free fan-out
        // marks n_fanned while it sits in the window above.
        dg.start_task(r);
        assert!(dg.finish_task_and_publish(r, Some(9)));
        sched.release_all();
        assert!(ha.join().expect("the phase-A finish thread must not panic"));
        assert!(
            !dg.is_green(n_fanned),
            "phase A: a fan-out mark landing in the finish window must survive it"
        );
        assert_eq!(
            dg.verify_green_transitive_with_dirty(n_fanned),
            Some(n_fanned),
            "phase A: the walk must report n_fanned ITSELF as the dirty node"
        );

        // -- Phase B: NO reverse edge, so the late re-check is the only repair --
        sched.arm(move |p| matches!(p, SchedulePoint::FinishWindow { node } if *node == n_late.0));
        let hb = shuttle::thread::spawn(move || {
            dg.start_task(n_late);
            dg.read(r);
            dg.finish_task_and_publish(n_late, Some(2))
        });
        sched.wait_parked();
        dg.start_task(r);
        assert!(dg.finish_task_and_publish(r, Some(10)));
        sched.release_all();
        assert!(hb.join().expect("the phase-B finish thread must not panic"));
        assert!(
            !dg.is_green(n_late),
            "phase B: n_late read R for the first time, so R's fan-out CANNOT \
             reach it - the late re-check is the ONLY thing that can restore \
             the bit, and it did not"
        );
        assert_eq!(
            dg.verify_green_transitive_with_dirty(n_late),
            Some(n_late),
            "phase B: n_late must be red specifically by the late re-check"
        );
    });
}
