use std::mem;
use std::ops::{Index, IndexMut};

use thiserror::Error;

pub(crate) struct Stack {
    entries: Vec<StackEntry>,
    depth_limit: usize,
}

#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub(crate) struct StackDepth {
    depth: usize,
}

pub(crate) struct StackEntry {
    cycle: bool,
}

#[derive(Debug, Error)]
pub(crate) enum StackError {
    #[error("stack depth limit reached")]
    Overflow,
}

impl Stack {
    pub(crate) fn new(depth_limit: usize) -> Self {
        Self {
            entries: vec![],
            depth_limit,
        }
    }

    pub(crate) fn push(&mut self) -> Result<StackDepth, StackError> {
        let depth = StackDepth {
            depth: self.entries.len(),
        };

        if depth.depth >= self.depth_limit {
            return Err(StackError::Overflow);
        }

        self.entries.push(StackEntry { cycle: false });
        Ok(depth)
    }

    pub(crate) fn pop(&mut self, depth: StackDepth) {
        assert_eq!(
            depth.depth + 1,
            self.entries.len(),
            "mismatched stack push/pop"
        );
        self.entries.pop();
    }
}

impl StackEntry {
    pub(crate) fn flag_cycle(&mut self) {
        self.cycle = true;
    }

    pub(crate) fn read_and_reset_cycle_flag(&mut self) -> bool {
        mem::replace(&mut self.cycle, false)
    }
}

impl Index<StackDepth> for Stack {
    type Output = StackEntry;

    fn index(&self, index: StackDepth) -> &Self::Output {
        &self.entries[index.depth]
    }
}

impl IndexMut<StackDepth> for Stack {
    fn index_mut(&mut self, index: StackDepth) -> &mut Self::Output {
        &mut self.entries[index.depth]
    }
}
