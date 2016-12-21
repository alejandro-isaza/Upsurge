//
//  Sequence.swift
//  Upsurge
//
//  Created by Adam Nemecek on 12/21/16.
//  Copyright © 2016 Venture Media Labs. All rights reserved.
//

import Foundation

extension Sequence {
    func all(predicate: (Iterator.Element) -> Bool) -> Bool {
        for e in self where !predicate(e) {
            return false
        }
        return true
    }
}

internal extension Collection {
    func indexIsValid(_ index: Index) -> Bool {
        return (startIndex..<endIndex).contains(index)
    }
}
