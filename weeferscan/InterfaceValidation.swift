//
//  InterfaceValidation.swift
//  weeferscan
//
//  Created by Winardi Susanto on 21/7/21.
//

import Foundation


protocol InterfaceValidation {

    func register(name: String , modelFace: ModelFace )

    func recognizeImage (bitmap: Any , getExtra: Bool) -> [ModelFace]

    func enableStatLogging(debug : Bool)

    func close()
}
